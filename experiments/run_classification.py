"""
Benchmark: Consolidated Image Classification & Trajectory Analysis
===================================================================
Unified ResNet-18 runner for CIFAR-10 and Fashion-MNIST across multi-seed evaluations.
Supports ReLU, GELU, Swish, Adaptive Swish, PReLU, PGELU, Static GoLU, and Adaptive Alpha-GoLU.
Includes strict softplus alpha constraints, robust statistics, and memory-decoupled architecture.

Author: Džana Kopić
Paper Reference: Gompertz Linear Units (Das et al., 2025)
"""

import inspect
import math
import sys
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.train_tuning import bf16_autocast, build_adamw_with_activation_groups, clip_activation_gradients, clamp_alpha_golu_modules, configure_benchmark_runtime, default_loader_kwargs, resolve_task_alpha_hparams


# ==========================================
# 0. Statistical Rigor & Utilities
# ==========================================
from utils.stats import compute_summary_statistics, calculate_p_value
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.overhead_tracker import OverheadTracker


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_benchmark_runtime()


# ==========================================
# 1. Activation Implementations
# ==========================================
class PGELU(nn.Module):
    """
    Parametric GELU: x * CDF(alpha * x) with softplus-constrained positive alpha parameter.
    """
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20.0 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    """
    Adaptive Swish (SiLU): x * sigmoid(beta * x)
    """
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        """Alias property for consistent parameter extraction across modules."""
        return self.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str, channels: int | None = None, alpha_layout: str = "channel") -> nn.Module:
    """Factory function for instantiating activation modules."""
    act_type = str(act_type).lower().strip()
    alpha_layout = str(alpha_layout).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type in ('adaptive_swish', 'swish_adaptive'):
        return AdaptiveSwish(init_beta=1.0)
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        if alpha_layout == "channel" and channels is not None:
            return AdaptiveAlphaGoLU(init_alpha=1.0, channels=channels)
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


# ==========================================
# 2. ResNet Architecture
# ==========================================
class ResNetBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, act_type: str = 'alpha_golu', alpha_layout: str = "channel"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        self.act1 = get_activation(act_type, channels=planes, alpha_layout=alpha_layout)
        self.act2 = get_activation(act_type, channels=planes, alpha_layout=alpha_layout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, act_type: str = 'alpha_golu', alpha_layout: str = "channel"):
        super().__init__()
        self.in_planes = 64
        self.act_type = act_type
        self.alpha_layout = alpha_layout
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = get_activation(act_type, channels=64, alpha_layout=alpha_layout)

        self.layer1 = self._make_layer(64, 2, stride=1, act_type=act_type)
        self.layer2 = self._make_layer(128, 2, stride=2, act_type=act_type)
        self.layer3 = self._make_layer(256, 2, stride=2, act_type=act_type)
        self.layer4 = self._make_layer(512, 2, stride=2, act_type=act_type)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, act_type: str):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_planes, planes, s, act_type, alpha_layout=self.alpha_layout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out, dim=[2, 3])  # Global Average Pooling
        out = self.linear(out)
        return out

    def extract_alphas(self) -> list:
        alphas = []
        for module in self.modules():
            if isinstance(module, (AdaptiveAlphaGoLU, AdaptiveSwish, PGELU)):
                val = module.alpha.detach().cpu().numpy().flatten()
                alphas.extend(val.tolist())
            elif isinstance(module, nn.PReLU):
                val = module.weight.detach().cpu().numpy().flatten()
                alphas.extend(val.tolist())
        return alphas


# ==========================================
# 3. Data Pipeline & Optimization Helpers
# ==========================================
def get_dataloaders(
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    seed: int = 42,
    root: str = "./data",
    val_split: float = 0.1,
    include_test: bool = True,
):
    dataset_name_lower = str(dataset_name).lower().strip()
    train_loader_kwargs = default_loader_kwargs()
    eval_loader_kwargs = default_loader_kwargs(num_workers=1)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset_name_lower == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test) if include_test else None
    elif dataset_name_lower == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform) if include_test else None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    val_size = int(len(trainset) * val_split)
    train_size = len(trainset) - val_size
    if val_size > 0:
        trainset, valset = random_split(trainset, [train_size, val_size], generator=g)
    else:
        valset = None

    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        **train_loader_kwargs,
    )
    valloader = None
    if valset is not None:
        valloader = DataLoader(
            valset,
            batch_size=256,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            **eval_loader_kwargs,
        )

    testloader = None
    if testset is not None:
        testloader = DataLoader(
            testset,
            batch_size=256,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            **eval_loader_kwargs,
        )
    return trainloader, valloader, testloader


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool = False) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs.float(), labels)
                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total += batch_size
                correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def summarize_model_overhead(model: nn.Module, train_samples: int, train_seconds: float) -> dict:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    alpha_param_count = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, AdaptiveAlphaGoLU)
        for parameter in module.parameters(recurse=False)
    )

    peak_cuda_memory_mb = None
    if torch.cuda.is_available():
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    samples_per_second = train_samples / max(train_seconds, 1e-12)
    seconds_per_sample = train_seconds / max(train_samples, 1)

    return {
        "total_params": int(total_params),
        "alpha_param_count": int(alpha_param_count),
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "train_samples": int(train_samples),
        "train_seconds": float(train_seconds),
        "seconds_per_sample": float(seconds_per_sample),
        "samples_per_second": float(samples_per_second),
    }


# ==========================================
# 4. Training & Benchmark Execution
# ==========================================
def train_single_seed(
    act_type: str,
    dataset_name: str = "cifar10",
    seed: int = 42,
    epochs: int = 10,
    device: torch.device = torch.device("cuda"),
    data_root: str = "./data",
    alpha_lr: float | None = None,
    val_split: float = 0.1,
    eval_split: str = "test",
    alpha_lr_scheduler: str = "none",
    alpha_layout: str = "channel",
    config_path: str | None = "configs/paper_benchmark.json",
    save_artifacts: bool = False,
    amp: bool = False,
    return_metrics: bool = False,
):
    reset_all_seeds(seed)
    trainloader, valloader, testloader = get_dataloaders(dataset_name, seed=seed, root=data_root, val_split=val_split, include_test=(eval_split == "test"))
    
    model = ResNet18(num_classes=10, act_type=act_type, alpha_layout=alpha_layout).to(device)
    overhead_tracker = OverheadTracker(task_name="classification", activation_name=act_type, model=model, device=device)
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "classification",
        alpha_lr,
        config_path=config_path,
    )
    
    optimizer, set_alpha_lr, act_params = build_adamw_with_activation_groups(
        model,
        base_lr=1e-3,
        base_weight_decay=5e-4,
        activation_lr=alpha_lr,
        activation_weight_decay=0.0,
        warmup_epochs=alpha_warmup_epochs,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    alpha_logger = AlphaTrajectoryLogger(model)
    train_start = time.perf_counter()
    epoch_seconds = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"

    def update_alpha_lr(epoch_index: int) -> float:
        if not act_params:
            return 0.0
        base_current = set_alpha_lr(epoch_index)
        if alpha_lr_scheduler == "cosine":
            denom = max(epochs - 1, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * epoch_index / denom))
            current_alpha_lr = base_current * scale
            optimizer.param_groups[1]["lr"] = current_alpha_lr
            return current_alpha_lr
        return base_current

    current_alpha_lr = update_alpha_lr(0)
    alpha_clamp_events = 0
    alpha_clamp_checks = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        current_alpha_lr = update_alpha_lr(epoch)
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            overhead_tracker.start_forward()
            with bf16_autocast(amp_enabled):
                outputs = model(inputs)
            overhead_tracker.end_forward(batch_size=inputs.size(0))
            loss = nn.CrossEntropyLoss()(outputs.float(), labels)
            overhead_tracker.start_backward()
            loss.backward()
            overhead_tracker.end_backward()
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks
        scheduler.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    train_seconds = time.perf_counter() - train_start

    eval_loader = valloader if eval_split == "val" else testloader
    if eval_loader is None:
        raise ValueError("Evaluation loader is not available for the requested eval_split")

    eval_loss, acc = evaluate_model(model, eval_loader, device, amp_enabled=amp_enabled)
    alphas = model.extract_alphas()
    overhead = overhead_tracker.save()

    if save_artifacts:
        run_dir = create_run_directory(
            str(PROJECT_ROOT / "outputs" / "runs" / "classification"),
            "classification",
            act_type,
            [seed],
        )
        write_json(
            run_dir / "results.json",
            {
                "task": "classification",
                "dataset_name": dataset_name,
                "data_root": data_root,
                "activation": act_type,
                "alpha_lr": alpha_lr,
                "seed": seed,
                "epochs": epochs,
                "eval_split": eval_split,
                "eval_loss": eval_loss,
                "accuracy": acc,
                "alpha_values": alphas,
                "alpha_history": alpha_logger.alpha_history,
                "train_seconds": train_seconds,
                "epoch_seconds": epoch_seconds,
                "alpha_lr_final": current_alpha_lr,
                "alpha_clamp_events": alpha_clamp_events,
                "alpha_clamp_checks": alpha_clamp_checks,
                **overhead,
            },
        )
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --dataset-name {dataset_name} --seeds {seed} --epochs {epochs}",
                task="classification",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "dataset_name": dataset_name,
                    "data_root": data_root,
                    "epochs": epochs,
                    "val_split": val_split,
                    "eval_split": eval_split,
                    "alpha_lr_scheduler": alpha_lr_scheduler,
                    "alpha_lr": alpha_lr,
                    "seed": seed,
                },
            ),
        )
        if alpha_logger.alpha_history:
            alpha_logger.plot_trajectories(str(run_dir / "alpha_trajectories.png"))
    
    if return_metrics:
        return {
            "accuracy": acc,
            "loss": eval_loss,
            "alphas": alphas,
            "eval_split": eval_split,
            "alpha_lr_final": current_alpha_lr,
                "alpha_layout": alpha_layout,
        }

    return acc, alphas


def run_benchmark(dataset_name: str = "cifar10", seeds: list = [42, 123, 999, 2024, 2025], epochs: int = 10, data_root: str = "./data", alpha_lr: float | None = None, val_split: float = 0.1, alpha_lr_scheduler: str = "none", alpha_layout: str = "channel", config_path: str | None = "configs/paper_benchmark.json", amp: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']
    results = {act: [] for act in activations}

    print(f"\n================ Running Unified ResNet-18 Benchmark on {dataset_name.upper()} (N={len(seeds)}) ================")
    
    for act in activations:
        print(f"\n--- Activation: {act.upper()} ---")
        for s in seeds:
            acc, alphas = train_single_seed(
                act,
                dataset_name=dataset_name,
                seed=s,
                epochs=epochs,
                device=device,
                data_root=data_root,
                alpha_lr=alpha_lr,
                val_split=val_split,
                eval_split="test",
                alpha_lr_scheduler=alpha_lr_scheduler,
                alpha_layout=alpha_layout,
                config_path=config_path,
                save_artifacts=True,
                amp=amp,
            )
            results[act].append(acc)
            if 'golu' in act and alphas:
                mean_alpha = np.mean(alphas)
                print(f"[{act.upper():<14} | Seed {s:4d}] Accuracy: {acc:.2f}% | Final Mean Alpha: {mean_alpha:.4f}")
            else:
                print(f"[{act.upper():<14} | Seed {s:4d}] Accuracy: {acc:.2f}%")

    print(f"\n================ {dataset_name.upper()} SUMMARY STATISTICS ================")
    for act, accs in results.items():
        stats_res = compute_summary_statistics(accs)
        print(f"  {act.upper():<14}: Mean = {stats_res['mean']:.2f}% ± {stats_res['std']:.2f}%")

    if 'golu_static' in results and 'alpha_golu' in results:
        p_val = calculate_p_value(results['golu_static'], results['alpha_golu'])
        print(f"\nStatistical Significance (Alpha-GoLU vs Static GoLU p-value): {p_val:.4f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, dataset_name: str = 'cifar10', epochs: int = 10, data_root: str = './data', alpha_lr: float | None = None, val_split: float = 0.1, alpha_lr_scheduler: str = "none", alpha_layout: str = "channel", config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc, _ = train_single_seed(act_type=activation, dataset_name=dataset_name, seed=seed, epochs=epochs, device=device, data_root=data_root, alpha_lr=alpha_lr, val_split=val_split, eval_split="test", alpha_lr_scheduler=alpha_lr_scheduler, alpha_layout=alpha_layout, config_path=config_path, save_artifacts=save_artifacts, amp=amp)
    return float(acc)


def run_alpha_layout_ablation(dataset_name: str = "cifar10", seeds: list[int] | None = None, epochs: int = 10, data_root: str = "./data", alpha_lr: float | None = None, val_split: float = 0.1, alpha_lr_scheduler: str = "none", config_path: str | None = "configs/paper_benchmark.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 999, 2024, 2025]
    layouts = ["layer", "channel"]

    print(f"\n================ Alpha Layout Ablation on {dataset_name.upper()} ================")
    for layout in layouts:
        scores = []
        print(f"\n--- Alpha Layout: {layout.upper()} ---")
        for seed in seeds:
            acc, _ = train_single_seed(
                act_type="alpha_golu",
                dataset_name=dataset_name,
                seed=seed,
                epochs=epochs,
                device=device,
                data_root=data_root,
                alpha_lr=alpha_lr,
                val_split=val_split,
                eval_split="test",
                alpha_lr_scheduler=alpha_lr_scheduler,
                alpha_layout=layout,
                config_path=config_path,
                save_artifacts=True,
            )
            scores.append(acc)
            print(f"Seed {seed} -> Accuracy: {acc:.2f}%")
        stats_res = compute_summary_statistics(scores)
        print(f"--> {layout.upper()} Mean Accuracy: {stats_res['mean']:.2f}% ± {stats_res['std']:.2f}%")


def run_alpha_lr_ablation(
    dataset_name: str = "cifar10",
    seeds: list[int] | None = None,
    epochs: int = 10,
    data_root: str = "./data",
    val_split: float = 0.1,
    alpha_lrs: list[float] | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42]
    alpha_lrs = alpha_lrs or [5e-4, 1e-3, 2e-3]

    print(f"\n================ Alpha LR Ablation on {dataset_name.upper()} ================")
    print(f"Seeds: {seeds}")
    print(f"Alpha LRs: {alpha_lrs}")

    summary = {}
    for alpha_lr in alpha_lrs:
        lr_key = f"alpha_lr_{alpha_lr:.0e}"
        summary[lr_key] = []
        print(f"\n--- Alpha LR: {alpha_lr:.1e} ---")
        for seed in seeds:
            metrics = train_single_seed(
                act_type="alpha_golu",
                dataset_name=dataset_name,
                seed=seed,
                epochs=epochs,
                device=device,
                data_root=data_root,
                alpha_lr=alpha_lr,
                val_split=val_split,
                eval_split="val",
                alpha_lr_scheduler="cosine",
                config_path="configs/paper_benchmark.json",
                save_artifacts=True,
                return_metrics=True,
            )
            summary[lr_key].append({"seed": seed, "accuracy": metrics["accuracy"], "loss": metrics["loss"], "alphas": metrics["alphas"]})
            mean_alpha = float(np.mean(metrics["alphas"])) if metrics["alphas"] else float("nan")
            print(f"Seed {seed} -> Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.2f}% | Final Mean Alpha: {mean_alpha:.4f}")

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Unified ResNet-18 benchmark")
    parser.add_argument("--dataset-name", type=str, default=None, choices=["cifar10", "fashion_mnist"], help="Dataset to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 2024, 2025], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data reserved for validation")
    parser.add_argument("--alpha-lr-scheduler", type=str, default="none", choices=["none", "cosine"], help="Scheduler for activation parameter LR")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--alpha-layout", type=str, default="channel", choices=["layer", "channel"], help="Alpha-GoLU parameter layout")
    parser.add_argument("--benchmark", action="store_true", help="Run the full benchmark sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    parser.add_argument("--alpha-layout-ablation", action="store_true", help="Compare layer-wise vs channel-wise Alpha-GoLU")
    parser.add_argument("--alpha-lr-ablation", action="store_true", help="Run a compact alpha LR sweep for Alpha-GoLU only")
    parser.add_argument("--alpha-lrs", type=float, nargs="+", default=None, help="Custom alpha LR sweep values for --alpha-lr-ablation")
    args = parser.parse_args()

    if args.alpha_layout_ablation:
        run_alpha_layout_ablation(dataset_name=args.dataset_name or "cifar10", seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, val_split=args.val_split, alpha_lr_scheduler=args.alpha_lr_scheduler, config_path=args.config)
    elif args.alpha_lr_ablation:
        run_alpha_lr_ablation(dataset_name=args.dataset_name or "cifar10", seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, val_split=args.val_split, alpha_lrs=args.alpha_lrs)
    elif args.benchmark:
        dataset_names = [args.dataset_name] if args.dataset_name else ["cifar10", "fashion_mnist"]
        for dataset_name in dataset_names:
            run_benchmark(dataset_name=dataset_name, seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, val_split=args.val_split, alpha_lr_scheduler=args.alpha_lr_scheduler, alpha_layout=args.alpha_layout, config_path=args.config, amp=args.amp)
    else:
        dataset_name = args.dataset_name or "cifar10"
        print(f"Running Classification Benchmark on {dataset_name.upper()}...")
        for seed in args.seeds:
            acc, _ = train_single_seed(
                act_type=args.activation,
                dataset_name=dataset_name,
                seed=seed,
                epochs=args.epochs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                data_root=args.data_root,
                alpha_lr=args.alpha_lr,
                val_split=args.val_split,
                eval_split="test",
                alpha_lr_scheduler=args.alpha_lr_scheduler,
                alpha_layout=args.alpha_layout,
                config_path=args.config,
                save_artifacts=True,
                amp=args.amp,
            )
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Accuracy: {acc:.2f}%")
