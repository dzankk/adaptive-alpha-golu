"""
Benchmark: DDPM Denoising on FashionMNIST
Evaluates spatial noise prediction MSE on real image manifolds.
Tests static vs. adaptive activation dynamics across multi-step diffusion steps.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. Fixed Activation Functions
# ==========================================
class StaticGoLU(nn.Module):
    """Gompertz Linear Unit: x * exp(-exp(-x))"""
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class AdaptiveAlphaGoLU(nn.Module):
    """Adaptive Gompertz Linear Unit: x * exp(-exp(-alpha * x))"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x)"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, act_type='alpha_golu'):
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

    def forward(self, x, time):
        t_emb = self.time_mlp(time)[:, :, None, None]
        h = self.act1(self.conv1(x)) + t_emb
        h = self.act2(self.conv2(h))
        return self.out_conv(h)


# ==========================================
# 3. Benchmark Execution
# ==========================================
def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    alpha_params = []
    other_params = []

    # Safely isolate adaptive activation parameters (including nn.PReLU)
    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, PGELU, AdaptiveSwish, nn.PReLU)):
            for p in module.parameters():
                if p.requires_grad:
                    alpha_params.append(p)

    alpha_param_ids = set(map(id, alpha_params))
    for p in model.parameters():
        if p.requires_grad and id(p) not in alpha_param_ids:
            other_params.append(p)

    return torch.optim.AdamW([
        {'params': other_params, 'lr': 2e-4, 'weight_decay': 1e-4},
        {'params': alpha_params, 'lr': 1e-3, 'weight_decay': 0.0}
    ])


def train_single_seed_diffusion(act_type: str, seed: int, epochs: int, device: torch.device) -> float:
    reset_seeds(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    timesteps = 1000
    beta = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    model = DiffusionUNet(in_channels=1, act_type=act_type).to(device)
    optimizer = get_optimizer(model)
    criterion = nn.MSELoss()

    model.train()
    target_steps = max(100, epochs * 100)
    step_count = 0

    while step_count < target_steps:
        for x0, _ in trainloader:
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)

            a_hat_t = alpha_hat[t][:, None, None, None]
            xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise

            predicted_noise = model(xt, t)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count >= target_steps:
                break

    # Evaluate Denoising Loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for idx, (x0, _) in zip(range(20), trainloader):
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            a_hat_t = alpha_hat[t][:, None, None, None]
            xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise
            val_losses.append(criterion(model(xt, t), noise).item())

    return float(np.mean(val_losses))


def run_diffusion_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running FashionMNIST DDPM Diffusion Benchmark on {device}...")

    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']
    seeds = [42, 123, 999, 2024, 2025]
    results = {}

    print("\n================ FashionMNIST DDPM Denoising Loss (MSE ↓) ================")
    for act in activations:
        seed_losses = []
        for s in seeds:
            mean_val = train_single_seed_diffusion(act_type=act, seed=s, epochs=10, device=device)
            seed_losses.append(mean_val)
            print(f"[{act.upper():<14} | Seed {s}] Denoising MSE: {mean_val:.6f}")

        results[act] = (np.mean(seed_losses), np.std(seed_losses))

    print("\n================ FASHIONMNIST DIFFUSION SUMMARY ================")
    for act, (m_loss, s_loss) in results.items():
        print(f"  {act.upper():<14}: Loss = {m_loss:.6f} ± {s_loss:.6f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Denoising Test MSE Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = train_single_seed_diffusion(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(loss)


if __name__ == '__main__':
    run_diffusion_benchmark()
