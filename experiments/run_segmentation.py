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
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, Dataset, random_split
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
from utils.train_tuning import bf16_autocast, clip_activation_gradients, clamp_alpha_golu_modules, configure_benchmark_runtime, default_loader_kwargs, overhead_tracking_enabled, resolve_task_alpha_hparams


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

        _replace_relu_modules(self.model.backbone, act_type)
        _replace_relu_modules(self.model.classifier, act_type)
        if getattr(self.model, "aux_classifier", None) is not None:
            _replace_relu_modules(self.model.aux_classifier, act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs["out"]


# ==========================================
# 3. Real Dataset & Metrics
# ==========================================
VOC_SEG_CLASSES = 21


class PascalVOCSegmentationDataset(Dataset):
    """Pascal VOC segmentation wrapper returning resized tensors and masks."""

    def __init__(self, root: str = './data', year: str = '2012', image_set: str = 'train', image_size: int = 256, download: bool = True):
        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_benchmark_runtime()


def train_single_seed_segmentation(act_type: str, seed: int, epochs: int, device: torch.device, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    set_seed(seed)
    
    full_dataset = PascalVOCSegmentationDataset(root=data_root, year='2012', image_set='train', image_size=256, download=True)
    val_dataset = PascalVOCSegmentationDataset(root=data_root, year='2012', image_set='val', image_size=256, download=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    # Use the actual Pascal VOC validation split for final reporting.
    val_ds = val_dataset

    loader_g = torch.Generator().manual_seed(seed)
    train_loader_kwargs = default_loader_kwargs()
    val_loader_kwargs = default_loader_kwargs(num_workers=1)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=loader_g, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, generator=loader_g, **val_loader_kwargs)

    model = ImageNetBackboneDeepLabV3(act_type=act_type, num_classes=VOC_SEG_CLASSES).to(device)
    overhead_tracker = OverheadTracker(task_name="segmentation", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    alpha_logger = AlphaTrajectoryLogger(model)
    _, _, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "segmentation",
        alpha_lr,
        config_path=config_path,
    )
    optimizer = get_optimizer(model, lr=base_lr)
    scheduler = PolynomialLR(optimizer, total_iters=max(int(epochs), 1), power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    epoch_seconds = []
    epoch_losses = []
    lr_history = []
    train_start = time.perf_counter()
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"
    alpha_clamp_events = 0
    alpha_clamp_checks = 0
    final_epoch_loss = None
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

    print(
        "[SEGMENTATION] Activation audit before training: "
        + ", ".join(activation_names[:12])
        + (" ..." if len(activation_names) > 12 else ""),
        flush=True,
    )

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
                "activation_modules": activation_names,
            },
        ),
    )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_history.append(current_lr)
        epoch_loss_total = 0.0
        epoch_batches = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            if overhead_tracker is not None:
                overhead_tracker.start_forward()
            with bf16_autocast(amp_enabled):
                out = model(x)
            if overhead_tracker is not None:
                overhead_tracker.end_forward(batch_size=x.size(0))
            loss = criterion(out.float(), y)
            if overhead_tracker is not None:
                overhead_tracker.start_backward()
            loss.backward()
            if overhead_tracker is not None:
                overhead_tracker.end_backward()
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks
        mean_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        epoch_seconds.append(time.perf_counter() - epoch_start)
        epoch_losses.append(mean_epoch_loss)
        scheduler.step()
        alpha_logger.step()
        final_epoch_loss = mean_epoch_loss
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
                    "lr_current": current_lr,
                    "alpha_clamp_events": alpha_clamp_events,
                    "alpha_clamp_checks": alpha_clamp_checks,
                    "alpha_history": alpha_logger.alpha_history,
                },
            )
        if not save_artifacts:
            print(f"[SEGMENTATION] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.4f} | lr={current_lr:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_start
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}

    model.eval()
    total_iou = 0.0
    total_samples = 0
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                batch_size = x.size(0)
                total_iou += compute_mIoU(out.float(), y, num_classes=VOC_SEG_CLASSES) * batch_size
                total_samples += batch_size

    miou = total_iou / total_samples if total_samples > 0 else 0.0

    write_json(
        run_dir / "results.json",
        {
            "task": "segmentation",
            "activation": act_type,
            "data_root": data_root,
            "seed": seed,
            "epochs": epochs,
            "base_lr": base_lr,
                "alpha_lr": alpha_lr,
            "optimizer": "SGD",
            "scheduler": "PolynomialLR",
            "miou": float(miou),
            "epoch_loss_history": epoch_losses,
            "lr_history": lr_history,
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
                "alpha_lr": alpha_lr,
                "seed": seed,
                "epochs": epochs,
                "progress_pct": 100.0,
                "epoch_loss": final_epoch_loss,
                "epoch_loss_history": epoch_losses,
                "epoch_seconds": epoch_seconds,
                "lr_history": lr_history,
                "miou": float(miou),
                "alpha_clamp_events": alpha_clamp_events,
                "alpha_clamp_checks": alpha_clamp_checks,
                "alpha_history": alpha_logger.alpha_history,
                **overhead,
            },
        )
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

    return miou


def run_segmentation_benchmark(seeds=None, epochs=30, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", amp: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds or [42, 123, 999, 2024, 2025]
    print(f"Running Segmentation Benchmark on {device} (N={len(seeds)})")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']

    for act_type in activations:
        scores = []
        for s in seeds:
            miou = train_single_seed_segmentation(act_type=act_type, seed=s, epochs=epochs, device=device, data_root=data_root, base_lr=base_lr, alpha_lr=alpha_lr, config_path=config_path, save_artifacts=True, amp=amp)
            scores.append(miou)
            print(f"Activation: {act_type.ljust(15)} | Seed {s} | Validation mIoU: {miou:.4f}")
        print(f"--> {act_type.upper()} Mean mIoU: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 30, data_root: str = './data', base_lr: float = 2e-2, alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    """Returns Mean Intersection over Union (mIoU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miou = train_single_seed_segmentation(act_type=activation, seed=seed, epochs=epochs, device=device, data_root=data_root, base_lr=base_lr, alpha_lr=alpha_lr, config_path=config_path, save_artifacts=save_artifacts, amp=amp)
    return float(miou)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Pascal VOC segmentation benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 2024, 2025], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Base SGD learning rate (paper default: 0.02; alternative: 0.01)")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    args = parser.parse_args()

    if args.benchmark:
        run_segmentation_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, base_lr=args.lr, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running Segmentation Benchmark on {device} (N={len(args.seeds)})")
        for seed in args.seeds:
            miou = train_and_eval(activation=args.activation, seed=seed, epochs=args.epochs, data_root=args.data_root, base_lr=args.lr, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp)
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Validation mIoU: {miou:.4f}")
