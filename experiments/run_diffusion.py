"""
Denoising Diffusion Probabilistic Model (DDPM) Benchmark
Evaluates noise residual prediction MSE across activation functions.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class StaticGoLU(nn.Module):
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))


def get_activation(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'swish_adaptive':
        return AdaptiveSwish(init_beta=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=0.50) if hasattr(AdaptiveAlphaGoLU, '__init__') and 'init_alpha' in AdaptiveAlphaGoLU.__init__.__code__.co_varnames else AdaptiveAlphaGoLU()
    raise ValueError(f"Unknown activation: {act_type}")


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
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, act_type='alpha_golu'):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(32),
            nn.Linear(32, 32),
            get_activation(act_type)
        )
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.act2 = get_activation(act_type)
        self.out_conv = nn.Conv2d(32, in_channels, kernel_size=1)

    def forward(self, x, time):
        t_emb = self.time_mlp(time)[:, :, None, None]
        h = self.act1(self.conv1(x)) + t_emb
        h = self.act2(self.conv2(h))
        return self.out_conv(h)


def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    act_params, reg_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name:
            act_params.append(param)
        else:
            reg_params.append(param)

    return torch.optim.AdamW([
        {'params': reg_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': act_params, 'weight_decay': 0.0, 'lr': lr * 5.0}
    ])


def run_diffusion_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Denoising Diffusion Benchmark on {device}...")

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    seeds = [42, 123]
    results = {}

    timesteps = 100
    beta = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print("\n================ DDPM Denoising Loss (MSE ↓) ================")
    for act in activations:
        losses = []
        for s in seeds:
            reset_seeds(s)
            model = DiffusionUNet(in_channels=1, act_type=act).to(device)
            optimizer = get_optimizer(model)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(300):
                x0 = torch.randn(32, 1, 28, 28, device=device)
                t = torch.randint(0, timesteps, (32,), device=device).long()
                noise = torch.randn_like(x0)

                a_hat_t = alpha_hat[t][:, None, None, None]
                xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise

                predicted_noise = model(xt, t)
                loss = criterion(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):
                    x0 = torch.randn(32, 1, 28, 28, device=device)
                    t = torch.randint(0, timesteps, (32,), device=device).long()
                    noise = torch.randn_like(x0)
                    a_hat_t = alpha_hat[t][:, None, None, None]
                    xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise
                    val_losses.append(criterion(model(xt, t), noise).item())

            mean_val_loss = np.mean(val_losses)
            losses.append(mean_val_loss)
            print(f"[{act.upper():<14} | Seed {s}] Denoising Val Loss: {mean_val_loss:.6f}")

        results[act] = (np.mean(losses), np.std(losses))

    print("\n================ DIFFUSION SUMMARY ================")
    for act, (mean_l, std_l) in results.items():
        print(f"  {act.upper():<14}: Loss = {mean_l:.6f} ± {std_l:.6f}")


if __name__ == '__main__':
    run_diffusion_benchmark()
