"""
Adversarial Security Benchmark (CIFAR-10)
Uses pre-trained ResNet-18 with activation swapping + fine-tuning.
Evaluates Clean, FGSM, and PGD robustness.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

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
    act_type = act_type.lower()
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


def replace_activations(model, act_type):
    """Recursively replaces ReLU activations in PyTorch models."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, get_activation(act_type))
        else:
            replace_activations(module, act_type)


def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, -1.0, 1.0)


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=7):
    orig = images.clone().detach()
    adv = images.clone().detach()

    for _ in range(iters):
        adv.requires_grad = True
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        step = adv + alpha * adv.grad.sign()
        eta = torch.clamp(step - orig, min=-epsilon, max=epsilon)
        adv = torch.clamp(orig + eta, min=-1.0, max=1.0).detach()

    return adv


def run_robustness_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Publication-Grade Adversarial Security Benchmark on {device}...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Scaled for ResNet feature maps
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    epsilon = 0.03

    print(f"\n================ CIFAR-10 ResNet-18 Security (Epsilon = {epsilon}) ================")
    
    for act in activations:
        reset_seeds(42)
        
        # Load Pre-trained Backbone
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        replace_activations(model, act)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Fine-tune model for 5 Epochs
        model.train()
        for epoch in range(5):
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # Evaluate Security Metrics
        model.eval()
        clean_accs, fgsm_accs, pgd_accs = [], [], []

        for i, (x, y) in enumerate(testloader):
            if i >= 15: # Evaluate on 15 test mini-batches (~2000 images)
                break
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                clean_accs.append((model(x).argmax(dim=1) == y).float().mean().item())

            x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
            with torch.no_grad():
                fgsm_accs.append((model(x_fgsm).argmax(dim=1) == y).float().mean().item())

            x_pgd = pgd_attack(model, x, y, epsilon=epsilon)
            with torch.no_grad():
                pgd_accs.append((model(x_pgd).argmax(dim=1) == y).float().mean().item())

        print(f"[{act.upper():<14}] Clean: {np.mean(clean_accs)*100:5.2f}% | FGSM: {np.mean(fgsm_accs)*100:5.2f}% | PGD: {np.mean(pgd_accs)*100:5.2f}%")


if __name__ == '__main__':
    run_robustness_benchmark()
