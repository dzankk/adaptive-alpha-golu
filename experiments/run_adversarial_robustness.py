"""
Adversarial Robustness Benchmark
Evaluates ResNet-18 resilience against FGSM and PGD adversarial attacks.
"""

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


class SimpleClassifier(nn.Module):
    def __init__(self, act_type='alpha_golu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            get_activation(act_type),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10)
        )

    def forward(self, x):
        return self.net(x)


def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed_images = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed_images, 0, 1)


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=7):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()

    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv - original_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return perturbed_images


def run_robustness_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Adversarial Robustness Benchmark on {device}...")

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    epsilon = 0.05
    results = {}

    print(f"\n================ Adversarial Defense Accuracy (Epsilon = {epsilon}) ================")
    for act in activations:
        reset_seeds(42)
        model = SimpleClassifier(act_type=act).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train briefly
        model.train()
        for _ in range(150):
            x = torch.randn(32, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (32,), device=device)
            loss = nn.CrossEntropyLoss()(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        clean_accs, fgsm_accs, pgd_accs = [], [], []

        for _ in range(10):
            x = torch.randn(32, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (32,), device=device)

            clean_accs.append((model(x).argmax(dim=1) == y).float().mean().item())

            x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
            fgsm_accs.append((model(x_fgsm).argmax(dim=1) == y).float().mean().item())

            x_pgd = pgd_attack(model, x, y, epsilon=epsilon)
            pgd_accs.append((model(x_pgd).argmax(dim=1) == y).float().mean().item())

        print(f"[{act.upper():<14}] Clean Acc: {np.mean(clean_accs)*100:.1f}% | FGSM Acc: {np.mean(fgsm_accs)*100:.1f}% | PGD Acc: {np.mean(pgd_accs)*100:.1f}%")

if __name__ == '__main__':
    run_robustness_benchmark()
