"""
Benchmark: Semantic Segmentation (DeepLabV3)
Measures pixel-level target segmentation performance (mIoU) across activation functions.
Uses a pretrained ImageNet ResNet backbone for standard fine-tuning and isolates
the benchmarked activation function while training with the paper's SGD + polynomial LR recipe.
"""

import math
import time
import sys
import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.overhead_tracker import OverheadTracker
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.train_tuning import bf16_autocast, clip_activation_gradients, clamp_alpha_golu_modules, configure_benchmark_runtime, default_loader_kwargs, overhead_tracking_enabled, resolve_task_alpha_hparams, set_activation_parameters_trainable, split_model_parameters


try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass


QUIET_STDOUT = os.getenv("ADAPTIVE_ALPHA_GOLU_QUIET_STDOUT", "1") == "1"


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================
# 1. Custom Activation Implementations
# ==========================================
class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x)"""
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    """Adaptive Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class ActivationStatsProbe:
    """Collects activation-input and activation-output statistics for a model epoch."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: list[str] = []
        self.stats: dict[str, dict[str, float]] = {}
        self.active = False
        self._register_hooks()

    def _track_module(self, module: nn.Module) -> bool:
        return isinstance(
            module,
            (
                nn.ReLU,
                nn.PReLU,
                nn.GELU,
                nn.SiLU,
                PGELU,
                SwishAdaptive,
                StaticGoLU,
                AdaptiveAlphaGoLU,
            ),
        )

    def _blank_stats(self) -> dict[str, float]:
        return {
            "batches": 0.0,
            "input_numel": 0.0,
            "input_sum": 0.0,
            "input_sumsq": 0.0,
            "input_min": float("inf"),
            "input_max": float("-inf"),
            "input_neg_frac": 0.0,
            "input_lt_neg2_frac": 0.0,
            "input_lt_neg4_frac": 0.0,
            "input_lt_neg8_frac": 0.0,
            "input_lt_neg88_frac": 0.0,
            "output_numel": 0.0,
            "output_sum": 0.0,
            "output_sumsq": 0.0,
            "output_min": float("inf"),
            "output_max": float("-inf"),
            "output_neg_frac": 0.0,
            "output_lt_neg2_frac": 0.0,
            "output_lt_neg4_frac": 0.0,
            "output_lt_neg8_frac": 0.0,
            "output_lt_neg88_frac": 0.0,
            "input_nonfinite": 0.0,
            "output_nonfinite": 0.0,
            "alpha_mean": float("nan"),
            "alpha_min": float("nan"),
            "alpha_max": float("nan"),
        }

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not name or name == "model":
                continue
            if not self._track_module(module):
                continue
            self.layer_names.append(name)
            self.stats[name] = self._blank_stats()
            self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def _make_hook(self, name: str, module: nn.Module):
        def hook(_module, inputs, output):
            if not self.active or not inputs:
                return

            x = inputs[0].detach().float()
            y = output.detach().float() if torch.is_tensor(output) else output[0].detach().float()
            record = self.stats[name]
            record["batches"] += 1.0

            def update(prefix: str, tensor: torch.Tensor) -> None:
                finite_mask = torch.isfinite(tensor)
                finite_tensor = tensor[finite_mask]
                record[f"{prefix}_numel"] += float(tensor.numel())
                record[f"{prefix}_nonfinite"] += float((~finite_mask).sum().item())
                if finite_tensor.numel() > 0:
                    record[f"{prefix}_sum"] += float(finite_tensor.sum().item())
                    record[f"{prefix}_sumsq"] += float((finite_tensor * finite_tensor).sum().item())
                    record[f"{prefix}_min"] = min(record[f"{prefix}_min"], float(finite_tensor.min().item()))
                    record[f"{prefix}_max"] = max(record[f"{prefix}_max"], float(finite_tensor.max().item()))
                    record[f"{prefix}_neg_frac"] += float((tensor < 0).float().mean().item())
                    record[f"{prefix}_lt_neg2_frac"] += float((tensor < -2).float().mean().item())
                    record[f"{prefix}_lt_neg4_frac"] += float((tensor < -4).float().mean().item())
                    record[f"{prefix}_lt_neg8_frac"] += float((tensor < -8).float().mean().item())
                    record[f"{prefix}_lt_neg88_frac"] += float((tensor < -88).float().mean().item())

            update("input", x)
            update("output", y)

            if hasattr(module, "get_alpha_val"):
                alpha = module.get_alpha_val().detach().float()
            elif hasattr(module, "alpha"):
                alpha_val = module.alpha
                alpha = alpha_val.detach().float() if torch.is_tensor(alpha_val) else torch.tensor(float(alpha_val), dtype=torch.float32)
            elif hasattr(module, "beta"):
                beta_val = module.beta
                alpha = beta_val.detach().float() if torch.is_tensor(beta_val) else torch.tensor(float(beta_val), dtype=torch.float32)
            else:
                alpha = None

            if alpha is not None and alpha.numel() > 0:
                record["alpha_mean"] = float(alpha.mean().item())
                record["alpha_min"] = float(alpha.min().item())
                record["alpha_max"] = float(alpha.max().item())

        return hook

    def start_epoch(self) -> None:
        self.active = True
        self.stats = {name: self._blank_stats() for name in self.layer_names}

    def stop_epoch(self) -> dict[str, dict[str, float]]:
        self.active = False
        return {name: self._finalize(record) for name, record in self.stats.items()}

    def _finalize(self, record: dict[str, float]) -> dict[str, float]:
        finalized = dict(record)
        for prefix in ("input", "output"):
            numel = max(finalized[f"{prefix}_numel"], 1.0)
            mean = finalized[f"{prefix}_sum"] / numel
            mean_sq = finalized[f"{prefix}_sumsq"] / numel
            variance = max(mean_sq - mean * mean, 0.0)
            finalized[f"{prefix}_mean"] = mean
            finalized[f"{prefix}_std"] = float(math.sqrt(variance))
            finalized[f"{prefix}_neg_frac"] = finalized[f"{prefix}_neg_frac"] / max(finalized["batches"], 1.0)
            finalized[f"{prefix}_lt_neg2_frac"] = finalized[f"{prefix}_lt_neg2_frac"] / max(finalized["batches"], 1.0)
            finalized[f"{prefix}_lt_neg4_frac"] = finalized[f"{prefix}_lt_neg4_frac"] / max(finalized["batches"], 1.0)
            finalized[f"{prefix}_lt_neg8_frac"] = finalized[f"{prefix}_lt_neg8_frac"] / max(finalized["batches"], 1.0)
            finalized[f"{prefix}_lt_neg88_frac"] = finalized[f"{prefix}_lt_neg88_frac"] / max(finalized["batches"], 1.0)
        return finalized

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def compact_report(self, epoch_stats: dict[str, dict[str, float]], max_layers: int = 8) -> list[str]:
        rows = []
        ranked = sorted(
            epoch_stats.items(),
            key=lambda item: (item[1].get("input_lt_neg4_frac", 0.0), item[1].get("input_neg_frac", 0.0)),
            reverse=True,
        )
        for name, record in ranked[:max_layers]:
            rows.append(
                (
                    f"{name}: in_mean={record.get('input_mean', 0.0):.4f} in_std={record.get('input_std', 0.0):.4f} "
                    f"in_neg={record.get('input_neg_frac', 0.0):.3f} in_lt_-4={record.get('input_lt_neg4_frac', 0.0):.3f} "
                    f"in_lt_-8={record.get('input_lt_neg8_frac', 0.0):.3f} in_lt_-88={record.get('input_lt_neg88_frac', 0.0):.3f} "
                    f"out_mean={record.get('output_mean', 0.0):.4f} out_std={record.get('output_std', 0.0):.4f} "
                    f"alpha={record.get('alpha_mean', float('nan')):.4f}"
                )
            )
        return rows


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type in ('adaptive_swish', 'swish_adaptive'):
        return SwishAdaptive(init_beta=1.0)
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(
    model: nn.Module,
    lr: float = 2e-2,
) -> optim.Optimizer:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=0.9,
        weight_decay=1e-4,
    )
    return optimizer


def compute_parameter_grad_norm(parameters: list[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    grad_norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            return float("nan")
        grad_norms.append(float(torch.linalg.vector_norm(grad.float(), ord=norm_type).item()))

    if not grad_norms:
        return 0.0

    stacked = torch.tensor(grad_norms, dtype=torch.float32)
    return float(torch.linalg.vector_norm(stacked, ord=norm_type).item())


# ==========================================
# 2. DeepLabV3 Architecture
# ==========================================
def _replace_relu_modules(module: nn.Module, act_type: str) -> None:
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.ReLU):
            setattr(module, child_name, get_activation(act_type))
        else:
            _replace_relu_modules(child_module, act_type)


class ImageNetBackboneDeepLabV3(nn.Module):
    def __init__(self, act_type: str, num_classes: int = 21):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        )

        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.model.classifier[-1].weight, mode="fan_out", nonlinearity="relu")
        if self.model.classifier[-1].bias is not None:
            nn.init.zeros_(self.model.classifier[-1].bias)

        _replace_relu_modules(self.model.classifier, act_type)
        if getattr(self.model, "aux_classifier", None) is not None:
            _replace_relu_modules(self.model.aux_classifier, act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs["out"]

    def forward_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def named_backbone_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if name.startswith("model.backbone.")]

    def named_head_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if not name.startswith("model.backbone.")]

    def set_backbone_base_trainable(self, trainable: bool) -> None:
        activation_ids = {id(parameter) for parameter in split_model_parameters(self)[1]}
        for name, parameter in self.named_parameters():
            if not name.startswith("model.backbone."):
                continue
            if id(parameter) in activation_ids:
                continue
            parameter.requires_grad_(trainable)


def _set_backbone_batchnorm_eval(model: nn.Module) -> None:
    backbone = getattr(model, "model", None)
    backbone = getattr(backbone, "backbone", None)
    if backbone is None:
        return

    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


# ==========================================
# 3. Real Dataset & Metrics
# ==========================================
VOC_SEG_CLASSES = 21
VOC_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class PascalVOCSegmentationDataset(Dataset):
    """Pascal VOC segmentation wrapper returning resized tensors and masks."""

    def __init__(self, root: str = './data', year: str = '2012', image_set: str = 'train', image_size: int = 256, train: bool = True, download: bool = True):
        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.train = bool(train)
        self.image_size = image_size

    def _apply_train_transforms(self, image, mask):
        scale = random.uniform(0.5, 2.0)
        width, height = image.size
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = TF.resize(image, (new_height, new_width), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_height, new_width), interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        pad_height = max(0, self.image_size - new_height)
        pad_width = max(0, self.image_size - new_width)
        if pad_height > 0 or pad_width > 0:
            image = TF.pad(image, [0, 0, pad_width, pad_height], fill=0)
            mask = TF.pad(mask, [0, 0, pad_width, pad_height], fill=255)

        image_width, image_height = image.size
        top = 0 if image_height == self.image_size else random.randint(0, image_height - self.image_size)
        left = 0 if image_width == self.image_size else random.randint(0, image_width - self.image_size)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        return image, mask

    def _apply_eval_transforms(self, image, mask):
        image = TF.resize(image, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        if self.train:
            image, mask = self._apply_train_transforms(image, mask)
        else:
            image, mask = self._apply_eval_transforms(image, mask)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask[mask == 255] = 255
        return image, mask


def compute_mIoU(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 21, ignore_index: int = 255) -> float:
    preds = preds.argmax(dim=1)
    valid_mask = targets != ignore_index
    iou_values = []

    for class_index in range(num_classes):
        pred_class = preds == class_index
        target_class = targets == class_index
        pred_class = pred_class & valid_mask
        target_class = target_class & valid_mask

        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()
        if union > 0:
            iou_values.append((intersection + 1e-6) / (union + 1e-6))

    return float(np.mean(iou_values)) if iou_values else 0.0


def accumulate_iou_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    intersections: np.ndarray,
    unions: np.ndarray,
    pred_counts: np.ndarray,
    target_counts: np.ndarray,
    num_classes: int = 21,
    ignore_index: int = 255,
) -> None:
    preds = logits.argmax(dim=1)
    valid_mask = targets != ignore_index

    for class_index in range(num_classes):
        pred_class = (preds == class_index) & valid_mask
        target_class = (targets == class_index) & valid_mask
        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()
        pred_total = pred_class.sum().item()
        target_total = target_class.sum().item()

        intersections[class_index] += float(intersection)
        unions[class_index] += float(union)
        pred_counts[class_index] += float(pred_total)
        target_counts[class_index] += float(target_total)


def compute_model_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    grad_norms = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            return float("nan")
        grad_norms.append(float(torch.linalg.vector_norm(grad.float(), ord=norm_type).item()))

    if not grad_norms:
        return 0.0

    stacked = torch.tensor(grad_norms, dtype=torch.float32)
    return float(torch.linalg.vector_norm(stacked, ord=norm_type).item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_benchmark_runtime()


def _resolve_segmentation_alpha_lr(
    alpha_lr: float | None,
    *,
    base_lr: float,
    alpha_lr_multiplier: float,
    config_path: str | None,
) -> float:
    default_alpha_lr = float(base_lr * float(alpha_lr_multiplier))
    resolved_alpha_lr, _, _ = resolve_task_alpha_hparams(
        "segmentation",
        alpha_lr,
        config_path=config_path,
        default_alpha_lr=default_alpha_lr,
    )
    return float(resolved_alpha_lr)


def train_single_seed_segmentation(act_type: str, seed: int, epochs: int, device: torch.device, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, alpha_lr_multiplier: float = 10.0, freeze_backbone_epochs: int = 2, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    set_seed(seed)
    
    full_dataset = PascalVOCSegmentationDataset(root=data_root, year='2012', image_set='train', image_size=256, train=True, download=True)
    val_dataset = PascalVOCSegmentationDataset(root=data_root, year='2012', image_set='val', image_size=256, train=False, download=True)
    train_ds = full_dataset
    val_ds = val_dataset

    loader_g = torch.Generator().manual_seed(seed)
    train_loader_kwargs = default_loader_kwargs()
    if int(train_loader_kwargs.get("num_workers", 0)) > 0:
        train_loader_kwargs["persistent_workers"] = True
    val_loader_kwargs = default_loader_kwargs(num_workers=0)
    val_loader_kwargs["pin_memory"] = True
    train_batch_size = 32
    eval_batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=loader_g, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, worker_init_fn=seed_worker, generator=loader_g, **val_loader_kwargs)

    model = ImageNetBackboneDeepLabV3(act_type=act_type, num_classes=VOC_SEG_CLASSES).to(device)
    overhead_tracker = OverheadTracker(task_name="segmentation", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    alpha_logger = AlphaTrajectoryLogger(model)
    _, _, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "segmentation",
        alpha_lr,
        config_path=config_path,
    )
    if str(act_type).lower().strip() == "alpha_golu":
        # More conservative alpha clipping to reduce early activation shock.
        alpha_grad_clip_norm = min(float(alpha_grad_clip_norm), 0.5)
    activation_params = split_model_parameters(model)[1]
    activation_param_ids = {id(parameter) for parameter in activation_params}
    backbone_base_params: list[torch.nn.Parameter] = []
    head_base_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if id(parameter) in activation_param_ids:
            continue
        if name.startswith("model.backbone."):
            backbone_base_params.append(parameter)
        else:
            head_base_params.append(parameter)

    activation_lr_value = _resolve_segmentation_alpha_lr(
        alpha_lr,
        base_lr=base_lr,
        alpha_lr_multiplier=alpha_lr_multiplier,
        config_path=config_path,
    )
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_base_params, "lr": float(base_lr), "weight_decay": 1e-4},
            {"params": head_base_params, "lr": float(base_lr), "weight_decay": 1e-4},
            # Keep activation params exempt from weight decay.
            {"params": activation_params, "lr": activation_lr_value, "weight_decay": 0.0},
        ],
        lr=float(base_lr),
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = PolynomialLR(optimizer, total_iters=max(int(epochs) * max(len(train_loader), 1), 1), power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    aux_loss_weight = 0.4
    epoch_seconds = []
    epoch_losses = []
    lr_history = []
    activation_lr_history = []
    grad_norm_history = []
    backbone_grad_norm_history = []
    head_grad_norm_history = []
    activation_grad_norm_history = []
    batch_grad_norm_history = []
    backbone_bn_eval_epoch_flags: list[bool] = []
    backbone_bn_layers = sum(
        1 for module in model.model.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    activation_stats_history: dict[str, dict[str, dict[str, float]]] = {}
    train_start = time.perf_counter()
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"
    alpha_clamp_events = 0
    alpha_clamp_checks = 0
    final_epoch_loss = None
    activation_probe = ActivationStatsProbe(model)
    run_dir = create_run_directory(
        str(PROJECT_ROOT / "outputs" / "runs" / "segmentation"),
        "segmentation",
        act_type,
        [seed],
    )
    progress_path = run_dir / "progress.json"

    activation_names = []
    for module_name, module in model.named_modules():
        if module_name in ("", "model"):
            continue
        if isinstance(module, (nn.ReLU, nn.PReLU)) or module.__class__.__name__ in {"PGELU", "SwishAdaptive", "AlphaGoLU", "StaticGoLU"}:
            activation_names.append(f"{module_name}:{module.__class__.__name__}")

    debug_log_lines = [
        "[SEGMENTATION] Activation audit before training: "
        + ", ".join(activation_names[:12])
        + (" ..." if len(activation_names) > 12 else "")
    ]

    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
            task="segmentation",
            seeds=[seed],
            activations=[act_type],
            extra_config={
                "data_root": data_root,
                "epochs": epochs,
                "base_lr": base_lr,
                "activation_lr": activation_lr_value,
                "alpha_lr": activation_lr_value,
                "alpha_lr_multiplier": alpha_lr_multiplier,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "batch_size": train_batch_size,
                "optimizer": "SGD",
                "scheduler": "PolynomialLR",
                "activation_modules": activation_names,
            },
        ),
    )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        freeze_backbone = epoch < max(int(freeze_backbone_epochs), 0)
        freeze_alpha_warmup = freeze_backbone and str(act_type).lower().strip() == "alpha_golu"
        for parameter in backbone_base_params:
            parameter.requires_grad_(not freeze_backbone)
        if freeze_alpha_warmup:
            set_activation_parameters_trainable(model, False)
        else:
            set_activation_parameters_trainable(model, True)
        model.train()
        backbone_bn_eval_applied = False
        if freeze_backbone:
            _set_backbone_batchnorm_eval(model)
            backbone_bn_eval_applied = True
        backbone_bn_eval_epoch_flags.append(backbone_bn_eval_applied)
        # Probe pre-warmup epochs and the first post-warmup epochs so alpha movement after
        # the freeze window is actually observable in the debug log/activation_stats_history.
        warmup_epochs = max(int(freeze_backbone_epochs), 0)
        probe_epoch = epoch < 2 or (warmup_epochs <= epoch < warmup_epochs + 2)
        if probe_epoch:
            activation_probe.start_epoch()
        current_lr = float(optimizer.param_groups[0]["lr"])
        current_activation_lr = float(optimizer.param_groups[-1]["lr"])
        epoch_loss_total = 0.0
        epoch_batches = 0
        epoch_grad_norm_total = 0.0
        epoch_backbone_grad_norm_total = 0.0
        epoch_head_grad_norm_total = 0.0
        epoch_activation_grad_norm_total = 0.0
        epoch_batch_grad_norms = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if overhead_tracker is not None:
                overhead_tracker.start_forward()
            with bf16_autocast(amp_enabled):
                outputs = model.forward_outputs(x)
            if overhead_tracker is not None:
                overhead_tracker.end_forward(batch_size=x.size(0))
            loss = criterion(outputs["out"].float(), y)
            if "aux" in outputs and outputs["aux"] is not None:
                aux_loss = criterion(outputs["aux"].float(), y)
                loss = loss + aux_loss_weight * aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite segmentation loss detected for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            if overhead_tracker is not None:
                overhead_tracker.start_backward()
            loss.backward()
            if overhead_tracker is not None:
                overhead_tracker.end_backward()
            grad_norm = compute_model_grad_norm(model)
            backbone_grad_norm = compute_parameter_grad_norm(backbone_base_params)
            head_grad_norm = compute_parameter_grad_norm(head_base_params)
            activation_grad_norm = compute_parameter_grad_norm(activation_params)
            if not (math.isfinite(grad_norm) and math.isfinite(backbone_grad_norm) and math.isfinite(head_grad_norm) and math.isfinite(activation_grad_norm)):
                raise RuntimeError(f"Non-finite gradient norm detected for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            epoch_grad_norm_total += grad_norm
            epoch_backbone_grad_norm_total += backbone_grad_norm
            epoch_head_grad_norm_total += head_grad_norm
            epoch_activation_grad_norm_total += activation_grad_norm
            epoch_batch_grad_norms.append(grad_norm)
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            scheduler.step()
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks
        mean_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        mean_epoch_grad_norm = epoch_grad_norm_total / max(epoch_batches, 1)
        mean_epoch_backbone_grad_norm = epoch_backbone_grad_norm_total / max(epoch_batches, 1)
        mean_epoch_head_grad_norm = epoch_head_grad_norm_total / max(epoch_batches, 1)
        mean_epoch_activation_grad_norm = epoch_activation_grad_norm_total / max(epoch_batches, 1)
        epoch_seconds.append(time.perf_counter() - epoch_start)
        epoch_losses.append(mean_epoch_loss)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        activation_lr_history.append(current_activation_lr)
        grad_norm_history.append(mean_epoch_grad_norm)
        backbone_grad_norm_history.append(mean_epoch_backbone_grad_norm)
        head_grad_norm_history.append(mean_epoch_head_grad_norm)
        activation_grad_norm_history.append(mean_epoch_activation_grad_norm)
        batch_grad_norm_history.append(epoch_batch_grad_norms)
        alpha_logger.step()
        final_epoch_loss = mean_epoch_loss
        activation_epoch_stats = activation_probe.stop_epoch() if probe_epoch else {}
        if probe_epoch:
            activation_stats_history[str(epoch + 1)] = activation_epoch_stats
            debug_log_lines.append(f"[SEGMENTATION][Epoch {epoch + 1}] Activation stats (top layers by negative tail):")
            for line in activation_probe.compact_report(activation_epoch_stats, max_layers=8):
                debug_log_lines.append(f"  {line}")
        if save_artifacts and progress_path is not None:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "task": "segmentation",
                    "data_root": data_root,
                    "activation": act_type,
                    "base_lr": base_lr,
                    "seed": seed,
                    "epochs": epochs,
                    "epoch": epoch + 1,
                    "progress_pct": float(((epoch + 1) / max(epochs, 1)) * 100.0),
                    "epoch_loss": mean_epoch_loss,
                    "epoch_loss_history": epoch_losses,
                    "epoch_seconds": epoch_seconds,
                    "lr_history": lr_history,
                    "activation_lr_history": activation_lr_history,
                    "lr_current": current_lr,
                    "activation_lr_current": current_activation_lr,
                    "grad_norm_history": grad_norm_history,
                    "backbone_grad_norm_history": backbone_grad_norm_history,
                    "head_grad_norm_history": head_grad_norm_history,
                    "activation_grad_norm_history": activation_grad_norm_history,
                    "batch_grad_norm_history": batch_grad_norm_history,
                    "grad_norm_epoch": mean_epoch_grad_norm,
                    "backbone_grad_norm_epoch": mean_epoch_backbone_grad_norm,
                    "head_grad_norm_epoch": mean_epoch_head_grad_norm,
                    "activation_grad_norm_epoch": mean_epoch_activation_grad_norm,
                    "activation_epoch_stats": activation_epoch_stats,
                    "activation_stats_history": activation_stats_history,
                    "freeze_backbone": bool(freeze_backbone),
                    "freeze_alpha_warmup": bool(freeze_alpha_warmup),
                    "backbone_batchnorm_eval_applied": bool(backbone_bn_eval_applied),
                    "backbone_batchnorm_layers": int(backbone_bn_layers),
                    "alpha_clamp_events": alpha_clamp_events,
                    "alpha_clamp_checks": alpha_clamp_checks,
                    "alpha_history": alpha_logger.alpha_history,
                },
            )
        if not save_artifacts and not QUIET_STDOUT:
            print(f"[SEGMENTATION] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.4f} | grad_norm={mean_epoch_grad_norm:.4f} | backbone_grad={mean_epoch_backbone_grad_norm:.4f} | head_grad={mean_epoch_head_grad_norm:.4f} | act_grad={mean_epoch_activation_grad_norm:.4f} | lr={lr_history[-1]:.6f} | act_lr={activation_lr_history[-1]:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_start
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}

    model.eval()
    class_intersections = np.zeros(VOC_SEG_CLASSES, dtype=np.float64)
    class_unions = np.zeros(VOC_SEG_CLASSES, dtype=np.float64)
    pred_pixel_counts = np.zeros(VOC_SEG_CLASSES, dtype=np.float64)
    target_pixel_counts = np.zeros(VOC_SEG_CLASSES, dtype=np.float64)
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                accumulate_iou_stats(
                    out.float(),
                    y,
                    class_intersections,
                    class_unions,
                    pred_pixel_counts,
                    target_pixel_counts,
                    num_classes=VOC_SEG_CLASSES,
                )

    per_class_iou: dict[str, float | None] = {}
    iou_values: list[float] = []
    for class_index, class_name in enumerate(VOC_CLASS_NAMES):
        union = float(class_unions[class_index])
        if union <= 0.0:
            per_class_iou[class_name] = None
            continue
        value = float((class_intersections[class_index] + 1e-6) / (union + 1e-6))
        per_class_iou[class_name] = value
        iou_values.append(value)

    total_pred_pixels = float(pred_pixel_counts.sum())
    total_target_pixels = float(target_pixel_counts.sum())
    pred_class_pixel_fraction = {
        class_name: float(pred_pixel_counts[idx] / total_pred_pixels) if total_pred_pixels > 0.0 else 0.0
        for idx, class_name in enumerate(VOC_CLASS_NAMES)
    }
    target_class_pixel_fraction = {
        class_name: float(target_pixel_counts[idx] / total_target_pixels) if total_target_pixels > 0.0 else 0.0
        for idx, class_name in enumerate(VOC_CLASS_NAMES)
    }

    dominant_pred_class = VOC_CLASS_NAMES[int(np.argmax(pred_pixel_counts))] if total_pred_pixels > 0.0 else "unknown"
    dominant_pred_fraction = float(np.max(pred_pixel_counts) / total_pred_pixels) if total_pred_pixels > 0.0 else 0.0

    miou = float(np.mean(iou_values)) if iou_values else 0.0

    write_json(
        run_dir / "results.json",
        {
            "task": "segmentation",
            "activation": act_type,
            "data_root": data_root,
            "seed": seed,
            "epochs": epochs,
            "base_lr": base_lr,
            "alpha_lr": activation_lr_value,
            "optimizer": "SGD",
            "scheduler": "PolynomialLR",
            "miou": float(miou),
            "per_class_iou": per_class_iou,
            "pred_class_pixel_fraction": pred_class_pixel_fraction,
            "target_class_pixel_fraction": target_class_pixel_fraction,
            "dominant_pred_class": dominant_pred_class,
            "dominant_pred_fraction": dominant_pred_fraction,
            "backbone_batchnorm_layers": int(backbone_bn_layers),
            "backbone_bn_eval_epoch_flags": backbone_bn_eval_epoch_flags,
            "freeze_backbone_epochs": int(max(int(freeze_backbone_epochs), 0)),
            "alpha_grad_clip_norm": float(alpha_grad_clip_norm),
            "activation_weight_decay": 0.0,
            "epoch_loss_history": epoch_losses,
            "lr_history": lr_history,
            "activation_lr_history": activation_lr_history,
            "grad_norm_history": grad_norm_history,
            "backbone_grad_norm_history": backbone_grad_norm_history,
            "head_grad_norm_history": head_grad_norm_history,
            "activation_grad_norm_history": activation_grad_norm_history,
            "batch_grad_norm_history": batch_grad_norm_history,
            "activation_stats_history": activation_stats_history,
            "alpha_clamp_events": alpha_clamp_events,
            "alpha_clamp_checks": alpha_clamp_checks,
            "alpha_history": alpha_logger.alpha_history,
            **overhead,
        },
    )
    if progress_path is not None:
        write_json(
            progress_path,
            {
                "status": "completed",
                "task": "segmentation",
                "data_root": data_root,
                "activation": act_type,
                "base_lr": base_lr,
                "alpha_lr": activation_lr_value,
                "seed": seed,
                "epochs": epochs,
                "progress_pct": 100.0,
                "epoch_loss": final_epoch_loss,
                "epoch_loss_history": epoch_losses,
                "epoch_seconds": epoch_seconds,
                "lr_history": lr_history,
                "activation_lr_history": activation_lr_history,
                "grad_norm_history": grad_norm_history,
                "backbone_grad_norm_history": backbone_grad_norm_history,
                "head_grad_norm_history": head_grad_norm_history,
                "activation_grad_norm_history": activation_grad_norm_history,
                "batch_grad_norm_history": batch_grad_norm_history,
                "activation_stats_history": activation_stats_history,
                "miou": float(miou),
                "per_class_iou": per_class_iou,
                "pred_class_pixel_fraction": pred_class_pixel_fraction,
                "target_class_pixel_fraction": target_class_pixel_fraction,
                "dominant_pred_class": dominant_pred_class,
                "dominant_pred_fraction": dominant_pred_fraction,
                "backbone_batchnorm_layers": int(backbone_bn_layers),
                "backbone_bn_eval_epoch_flags": backbone_bn_eval_epoch_flags,
                "freeze_backbone_epochs": int(max(int(freeze_backbone_epochs), 0)),
                "alpha_grad_clip_norm": float(alpha_grad_clip_norm),
                "activation_weight_decay": 0.0,
                "alpha_clamp_events": alpha_clamp_events,
                "alpha_clamp_checks": alpha_clamp_checks,
                "alpha_history": alpha_logger.alpha_history,
                **overhead,
            },
        )
    activation_probe.close()
    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
            task="segmentation",
            seeds=[seed],
            activations=[act_type],
            extra_config={
                "data_root": data_root,
                "epochs": epochs,
                "base_lr": base_lr,
                "optimizer": "SGD",
                "scheduler": "PolynomialLR",
            },
        ),
    )
    if debug_log_lines:
        (run_dir / "debug_metrics.log").write_text("\n".join(debug_log_lines) + "\n", encoding="utf-8")

    return miou


def run_segmentation_benchmark(seeds=None, epochs=30, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, alpha_lr_multiplier: float = 10.0, freeze_backbone_epochs: int = 2, config_path: str | None = "configs/paper_benchmark.json", amp: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds or [42, 123, 999]
    print(f"Running Segmentation Benchmark on {device} (N={len(seeds)})")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']

    for act_type in activations:
        scores = []
        for s in seeds:
            miou = train_single_seed_segmentation(act_type=act_type, seed=s, epochs=epochs, device=device, data_root=data_root, base_lr=base_lr, alpha_lr=alpha_lr, alpha_lr_multiplier=alpha_lr_multiplier, freeze_backbone_epochs=freeze_backbone_epochs, config_path=config_path, save_artifacts=True, amp=amp)
            scores.append(miou)
            print(f"Activation: {act_type.ljust(15)} | Seed {s} | Validation mIoU: {miou:.4f}")
        print(f"--> {act_type.upper()} Mean mIoU: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 30, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, alpha_lr_multiplier: float = 10.0, freeze_backbone_epochs: int = 2, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    """Returns Mean Intersection over Union (mIoU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miou = train_single_seed_segmentation(act_type=activation, seed=seed, epochs=epochs, device=device, data_root=data_root, base_lr=base_lr, alpha_lr=alpha_lr, alpha_lr_multiplier=alpha_lr_multiplier, freeze_backbone_epochs=freeze_backbone_epochs, config_path=config_path, save_artifacts=save_artifacts, amp=amp)
    return float(miou)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Pascal VOC segmentation benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Base SGD learning rate (paper default: 0.02; alternative: 0.01)")
    parser.add_argument("--alpha-lr-multiplier", type=float, default=10.0, help="Multiplier applied to base LR for activation parameters when alpha_lr is not set explicitly")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2, help="Number of initial epochs to freeze backbone non-activation weights")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    args = parser.parse_args()

    if args.benchmark:
        run_segmentation_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, base_lr=args.lr, alpha_lr=args.alpha_lr, alpha_lr_multiplier=args.alpha_lr_multiplier, freeze_backbone_epochs=args.freeze_backbone_epochs, config_path=args.config, amp=args.amp)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running Segmentation Benchmark on {device} (N={len(args.seeds)})")
        for seed in args.seeds:
            miou = train_and_eval(activation=args.activation, seed=seed, epochs=args.epochs, data_root=args.data_root, base_lr=args.lr, alpha_lr=args.alpha_lr, alpha_lr_multiplier=args.alpha_lr_multiplier, freeze_backbone_epochs=args.freeze_backbone_epochs, config_path=args.config, amp=args.amp)
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Validation mIoU: {miou:.4f}")
