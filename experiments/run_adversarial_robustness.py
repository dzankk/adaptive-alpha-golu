"""
Benchmark: Corruption Robustness on CIFAR-10 (ResNet-18)
========================================================
Evaluates clean accuracy vs. lightweight corruption robustness across
ReLU, GELU, Swish, PReLU, PGELU, Static GoLU, Adaptive Alpha-GoLU, and
Adaptive Swish using Gaussian noise, shot noise, and blur perturbations.
Includes deterministic evaluation and CUDA seed resetting.
"""

import math
import inspect
import sys
import random
import time
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.overhead_tracker import OverheadTracker
from utils.train_tuning import bf16_autocast, build_adamw_with_activation_groups, clip_activation_gradients, clamp_alpha_golu_modules, compute_model_grad_norm, configure_benchmark_runtime, default_loader_kwargs, overhead_tracking_enabled, resolve_task_alpha_hparams


def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_benchmark_runtime()


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================
# 1. Custom Activation Implementations
# ==========================================
class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x)"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        return nn.functional.softplus(self.raw_alpha)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    """Parametric Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


AdaptiveSwish = SwishAdaptive


# ==========================================
# 2. Helpers & Factory
# ==========================================
def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.00)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.00)
    elif act_type in ('swish_adaptive', 'adaptive_swish'):
        return SwishAdaptive(init_beta=1.00)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    alpha_lr: float | None = None,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 1,
) -> tuple[optim.Optimizer, Callable[[int], float], list[nn.Parameter]]:
    return build_adamw_with_activation_groups(
        model,
        base_lr=lr,
        base_weight_decay=weight_decay,
        activation_lr=alpha_lr,
        activation_weight_decay=0.0,
        warmup_epochs=warmup_epochs,
    )


# ==========================================
# 3. ResNet Architecture
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = get_activation(act_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, act_type='relu', num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.act_type = act_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = get_activation(act_type)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.act_type))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out, dim=[2, 3])
        out = self.linear(out)
        return out


# ==========================================
# 4. Corruption Evaluation Utilities
# ==========================================
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

CORRUPTION_SUITE = {
    "gaussian_noise": {"sigma": 0.1},
    "shot_noise": {"severity": 0.03},
    "blur": {"kernel_size": 5, "sigma": 1.0},
}


def get_normalized_bounds(device):
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    return min_val, max_val


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = CIFAR_MEAN.to(images.device)
    std = CIFAR_STD.to(images.device)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = CIFAR_MEAN.to(images.device)
    std = CIFAR_STD.to(images.device)
    return (images - mean) / std


def apply_gaussian_noise(images: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    return torch.clamp(images + torch.randn_like(images) * sigma, 0.0, 1.0)


def apply_shot_noise(images: torch.Tensor, severity: float = 0.03) -> torch.Tensor:
    scale = max(1.0, 1.0 / max(severity, 1e-6))
    noisy = torch.poisson(images.clamp(0.0, 1.0) * scale) / scale
    return torch.clamp(noisy, 0.0, 1.0)


def apply_blur(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    blurred = [transforms.functional.gaussian_blur(image, kernel_size=kernel_size, sigma=[sigma, sigma]) for image in images]
    return torch.stack(blurred, dim=0)


def corrupt_batch(images: torch.Tensor, corruption_name: str) -> torch.Tensor:
    pixel_images = denormalize_images(images)
    if corruption_name == "gaussian_noise":
        corrupted = apply_gaussian_noise(pixel_images, sigma=CORRUPTION_SUITE[corruption_name]["sigma"])
    elif corruption_name == "shot_noise":
        corrupted = apply_shot_noise(pixel_images, severity=CORRUPTION_SUITE[corruption_name]["severity"])
    elif corruption_name == "blur":
        params = CORRUPTION_SUITE[corruption_name]
        corrupted = apply_blur(pixel_images, kernel_size=params["kernel_size"], sigma=params["sigma"])
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    return normalize_images(corrupted)


# ==========================================
# 5. Benchmark Execution Functions
# ==========================================
def train_single_seed_robustness(
    act_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/paper_benchmark.json",
    save_artifacts: bool = False,
    amp: bool = False,
) -> tuple[float, float]:
    reset_all_seeds(seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    loader_g = torch.Generator().manual_seed(seed)
    train_loader_kwargs = default_loader_kwargs()
    test_loader_kwargs = default_loader_kwargs(num_workers=1)
    train_loader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=loader_g,
        **train_loader_kwargs,
    )
    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=loader_g,
        **test_loader_kwargs,
    )

    model = ResNet18(act_type=act_type).to(device)
    overhead_tracker = OverheadTracker(task_name="robustness", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "robustness",
        alpha_lr,
        config_path=config_path,
    )
    optimizer, set_alpha_lr, act_params = get_optimizer(model, lr=1e-3, alpha_lr=alpha_lr, warmup_epochs=alpha_warmup_epochs)
    criterion = nn.CrossEntropyLoss()
    alpha_logger = AlphaTrajectoryLogger(model)
    epoch_seconds = []
    epoch_losses = []
    lr_history = []
    grad_norm_history = []
    train_start = time.perf_counter()
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"
    alpha_clamp_events = 0
    alpha_clamp_checks = 0
    run_dir = None
    progress_path = None
    if save_artifacts:
        run_dir = create_run_directory(
            str(PROJECT_ROOT / "outputs" / "runs" / "adversarial_robustness"),
            "robustness",
            act_type,
            [seed],
        )
        progress_path = run_dir / "progress.json"
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="robustness",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "epochs": epochs,
                    "activation": act_type,
                    "data_root": data_root,
                    "seed": seed,
                    "evaluation": {
                        "suite": "corruption",
                        "corruptions": CORRUPTION_SUITE,
                    },
                },
            ),
        )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        current_alpha_lr = set_alpha_lr(epoch)
        epoch_loss_total = 0.0
        epoch_batches = 0
        epoch_grad_norm_total = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            if overhead_tracker is not None:
                overhead_tracker.start_forward()
            with bf16_autocast(amp_enabled):
                outputs = model(inputs)
            if overhead_tracker is not None:
                overhead_tracker.end_forward(batch_size=inputs.size(0))
            loss = criterion(outputs.float(), targets)
            if overhead_tracker is not None:
                overhead_tracker.start_backward()
            loss.backward()
            if overhead_tracker is not None:
                overhead_tracker.end_backward()
            grad_norm = compute_model_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm detected for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            epoch_grad_norm_total += grad_norm
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()
        mean_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        mean_epoch_grad_norm = epoch_grad_norm_total / max(epoch_batches, 1)
        epoch_losses.append(mean_epoch_loss)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        grad_norm_history.append(mean_epoch_grad_norm)
        if save_artifacts and progress_path is not None:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "task": "robustness",
                    "data_root": data_root,
                    "activation": act_type,
                    "alpha_lr": alpha_lr if alpha_lr is not None else 1e-3,
                    "seed": seed,
                    "epochs": epochs,
                    "epoch": epoch + 1,
                    "progress_pct": float(((epoch + 1) / max(epochs, 1)) * 100.0),
                    "epoch_loss": mean_epoch_loss,
                    "epoch_loss_history": epoch_losses,
                    "epoch_seconds": epoch_seconds,
                    "lr_history": lr_history,
                    "grad_norm_history": grad_norm_history,
                    "grad_norm_epoch": mean_epoch_grad_norm,
                    "alpha_lr_final": current_alpha_lr,
                    "alpha_clamp_events": alpha_clamp_events,
                    "alpha_clamp_checks": alpha_clamp_checks,
                    "alpha_history": alpha_logger.alpha_history,
                },
            )
        if not save_artifacts:
            print(f"[ROBUSTNESS] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.4f} | alpha_lr={current_alpha_lr:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_start

    model.eval()
    clean_correct, total = 0, 0
    corruption_correct = {name: 0 for name in CORRUPTION_SUITE}

    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.no_grad():
            with bf16_autocast(amp_enabled):
                clean_correct += (model(images).argmax(1) == labels).sum().item()

        for corruption_name in CORRUPTION_SUITE:
            corrupted_images = corrupt_batch(images, corruption_name)
            with torch.no_grad():
                with bf16_autocast(amp_enabled):
                    corruption_correct[corruption_name] += (model(corrupted_images).argmax(1) == labels).sum().item()

        total += labels.size(0)

    clean_acc = (clean_correct / total) * 100.0 if total > 0 else 0.0
    corruption_accs = {
        name: (correct / total) * 100.0 if total > 0 else 0.0
        for name, correct in corruption_correct.items()
    }
    corruption_acc = float(np.mean(list(corruption_accs.values()))) if corruption_accs else 0.0
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}

    if save_artifacts:
        write_json(
            run_dir / "results.json",
            {
                "activation": act_type,
                "data_root": data_root,
                "seed": seed,
                "epochs": epochs,
                "alpha_lr": alpha_lr if alpha_lr is not None else 1e-3,
                "alpha_lr_final": current_alpha_lr if act_params else None,
                "epoch_loss_history": epoch_losses,
                "lr_history": lr_history,
                "grad_norm_history": grad_norm_history,
                "clean_acc": clean_acc,
                "corruption_acc": corruption_acc,
                **{f"{name}_acc": value for name, value in corruption_accs.items()},
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
                    "task": "robustness",
                    "data_root": data_root,
                    "activation": act_type,
                    "alpha_lr": alpha_lr if alpha_lr is not None else 1e-3,
                    "seed": seed,
                    "epochs": epochs,
                    "progress_pct": 100.0,
                    "alpha_lr_final": current_alpha_lr if act_params else None,
                    "epoch_loss_history": epoch_losses,
                    "lr_history": lr_history,
                    "grad_norm_history": grad_norm_history,
                    "clean_acc": clean_acc,
                    "corruption_acc": corruption_acc,
                    **{f"{name}_acc": value for name, value in corruption_accs.items()},
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
                task="robustness",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "epochs": epochs,
                    "activation": act_type,
                    "data_root": data_root,
                    "seed": seed,
                    "attack": {
                        "eps": 8 / 255,
                        "alpha": 2 / 255,
                        "iters": 10,
                    },
                },
            ),
        )
    return clean_acc, corruption_acc
def run_benchmark(seeds=None, epochs: int = 10, data_root: str = './data', amp: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Corruption Robustness Benchmark on {device}")
    activations = ['relu', 'gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu', 'swish_adaptive']
    seeds = seeds or [42, 123, 999]

    for act_type in activations:
        print(f"\n--- Activation: {act_type.upper()} ---")
        for seed in seeds:
            clean_acc, corruption_acc = train_single_seed_robustness(
                act_type=act_type,
                seed=seed,
                epochs=epochs,
                device=device,
                data_root=data_root,
                save_artifacts=True,
                amp=amp,
            )
            print(f"Seed {seed} -> Clean Acc: {clean_acc:.2f}% | Corruption Acc: {corruption_acc:.2f}%")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10, data_root: str = './data', alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    """Returns mean corruption accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, corruption_acc = train_single_seed_robustness(
        act_type=activation,
        seed=seed,
        epochs=epochs,
        device=device,
        data_root=data_root,
        alpha_lr=alpha_lr,
        config_path=config_path,
        save_artifacts=save_artifacts,
        amp=amp,
    )
    return float(corruption_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CIFAR-10 adversarial robustness benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, amp=args.amp)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running Corruption Robustness Benchmark on {device}")
        for seed in args.seeds:
            corruption_acc = train_and_eval(
                activation=args.activation,
                seed=seed,
                epochs=args.epochs,
                data_root=args.data_root,
                alpha_lr=args.alpha_lr,
                config_path=args.config,
                save_artifacts=True,
                amp=args.amp,
            )
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Corruption Acc: {corruption_acc:.2f}%")
