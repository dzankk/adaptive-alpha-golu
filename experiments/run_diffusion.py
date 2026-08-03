"""
Benchmark: DDPM Denoising on CelebA
Evaluates spatial noise prediction MSE on real image manifolds.
Tests static vs. adaptive activation dynamics across multi-step diffusion steps.
Fully audited for parameter constraints, proper iteration counts, and deterministic evaluation.
"""

import math
import time
import sys
import random
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.overhead_tracker import OverheadTracker
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.train_tuning import build_adamw_with_activation_groups, clip_activation_gradients, default_loader_kwargs, resolve_task_alpha_hparams


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. Custom Activation Definitions
# ==========================================
class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x) with positive alpha constraint."""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    """Parametric Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type in ('swish_adaptive', 'adaptive_swish'):
        return AdaptiveSwish(init_beta=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    raise ValueError(f"Unknown activation: {act_type}")


# ==========================================
# 2. Diffusion Architecture
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, act_type='alpha_golu'):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 64),
            get_activation(act_type)
        )
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act2 = get_activation(act_type)
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)[:, :, None, None]
        h = self.act1(self.conv1(x)) + t_emb
        h = self.act2(self.conv2(h))
        return self.out_conv(h)


# ==========================================
# 3. Benchmark Execution Functions
# ==========================================
def get_optimizer(model: nn.Module, lr: float = 2e-4, alpha_lr: float | None = None, warmup_epochs: int = 1) -> tuple[torch.optim.Optimizer, Callable[[int], float], list[nn.Parameter]]:
    return build_adamw_with_activation_groups(
        model,
        base_lr=lr,
        base_weight_decay=1e-4,
        activation_lr=alpha_lr,
        activation_weight_decay=0.0,
        warmup_epochs=warmup_epochs,
    )


def train_single_seed_diffusion(act_type: str, seed: int, epochs: int, device: torch.device, data_root: str = './data', alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json") -> float:
    reset_seeds(seed)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = torchvision.datasets.CelebA(root=data_root, split='train', target_type='attr', download=True, transform=transform)
    test_dataset = torchvision.datasets.CelebA(root=data_root, split='valid', target_type='attr', download=True, transform=transform)

    loader_g = torch.Generator().manual_seed(seed)
    eval_g = torch.Generator().manual_seed(seed + 999)

    loader_kwargs = default_loader_kwargs()
    trainloader = DataLoader(
        full_dataset,
        batch_size=128,
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed((seed + worker_id) % 2**32),
        generator=loader_g,
        **loader_kwargs,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        worker_init_fn=lambda worker_id: np.random.seed((seed + 999 + worker_id) % 2**32),
        generator=eval_g,
        **loader_kwargs,
    )

    timesteps = 1000
    beta = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    model = DiffusionUNet(in_channels=3, act_type=act_type).to(device)
    overhead_tracker = OverheadTracker(task_name="diffusion", activation_name=act_type, model=model, device=device)
    alpha_logger = AlphaTrajectoryLogger(model)
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "diffusion",
        alpha_lr,
        config_path=config_path,
    )
    optimizer, set_alpha_lr, act_params = get_optimizer(model, alpha_lr=alpha_lr, warmup_epochs=alpha_warmup_epochs)
    criterion = nn.MSELoss()
    epoch_seconds = []
    train_start = time.perf_counter()

    # Complete Epoch Training Run
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        current_alpha_lr = set_alpha_lr(epoch)
        for x0, _ in trainloader:
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)

            a_hat_t = alpha_hat[t][:, None, None, None]
            xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise

            overhead_tracker.start_forward()
            predicted_noise = model(xt, t)
            overhead_tracker.end_forward(batch_size=x0.size(0))
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            overhead_tracker.start_backward()
            loss.backward()
            overhead_tracker.end_backward()
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()
        train_seconds = time.perf_counter() - train_start

    model.eval()
    val_losses = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for x0, _ in testloader:
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device, generator=eval_g).long()
            noise = torch.randn(x0.shape, device=device, generator=eval_g)

            a_hat_t = alpha_hat[t][:, None, None, None]
            xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise

            pred_noise = model(xt, t)
            batch_loss = criterion(pred_noise, noise).item()
            batch_size = x0.size(0)
            val_losses.append(batch_loss)
            total_loss += batch_loss * batch_size
            total_examples += batch_size

    overhead = overhead_tracker.save()
    loss = float(total_loss / total_examples) if total_examples > 0 else float(np.mean(val_losses)) if val_losses else 0.0

    run_dir = create_run_directory(
        str(PROJECT_ROOT / "outputs" / "runs" / "diffusion"),
        "diffusion",
        act_type,
        [seed],
    )
    write_json(
        run_dir / "results.json",
        {
            "task": "diffusion",
            "activation": act_type,
            "data_root": data_root,
            "seed": seed,
            "epochs": epochs,
            "alpha_lr": alpha_lr if alpha_lr is not None else 2e-4,
            "alpha_lr_final": current_alpha_lr if act_params else None,
            "mse": float(loss),
            "alpha_history": alpha_logger.alpha_history,
            **overhead,
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
            task="diffusion",
            seeds=[seed],
            activations=[act_type],
            extra_config={
                "data_root": data_root,
                "epochs": epochs,
            },
        ),
    )
    return loss


def run_diffusion_benchmark(seeds=None, epochs: int = 10, data_root: str = './data', alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running CelebA DDPM Diffusion Benchmark on {device}...")

    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']
    seeds = seeds or [42, 123, 999, 2024, 2025]
    results = {}

    print("\n================ CelebA DDPM Denoising Loss (MSE ↓) ================")
    for act in activations:
        seed_losses = []
        for s in seeds:
            mean_val = train_single_seed_diffusion(act_type=act, seed=s, epochs=epochs, device=device, data_root=data_root, alpha_lr=alpha_lr, config_path=config_path)
            seed_losses.append(mean_val)
            print(f"[{act.upper():<14} | Seed {s}] Denoising MSE: {mean_val:.6f}")

        results[act] = (np.mean(seed_losses), np.std(seed_losses))

    print("\n================ CELEBA DIFFUSION SUMMARY ================")
    for act, (m_loss, s_loss) in results.items():
        print(f"  {act.upper():<14}: Loss = {m_loss:.6f} ± {s_loss:.6f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10, data_root: str = './data', alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json") -> float:
    """Returns Denoising Test MSE Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = train_single_seed_diffusion(act_type=activation, seed=seed, epochs=epochs, device=device, data_root=data_root, alpha_lr=alpha_lr, config_path=config_path)
    return float(loss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CelebA diffusion benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 2024, 2025], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    args = parser.parse_args()

    if args.benchmark:
        run_diffusion_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running CelebA DDPM Diffusion Benchmark on {device}...")
        for seed in args.seeds:
            loss = train_and_eval(activation=args.activation, seed=seed, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config)
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Denoising MSE: {loss:.6f}")
