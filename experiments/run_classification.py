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
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU


# ==========================================
# 0. Statistical Rigor & Utilities
# ==========================================
from utils.stats import compute_summary_statistics, calculate_p_value


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_activation(act_type: str) -> nn.Module:
    """Factory function for instantiating activation modules."""
    act_type = str(act_type).lower().strip()
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
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


# ==========================================
# 2. ResNet Architecture
# ==========================================
class ResNetBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, act_type: str = 'alpha_golu'):
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

        self.act1 = get_activation(act_type)
        self.act2 = get_activation(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, act_type: str = 'alpha_golu'):
        super().__init__()
        self.in_planes = 64
        self.act_type = act_type
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = get_activation(act_type)

        self.layer1 = self._make_layer(64, 2, stride=1, act_type=act_type)
        self.layer2 = self._make_layer(128, 2, stride=2, act_type=act_type)
        self.layer3 = self._make_layer(256, 2, stride=2, act_type=act_type)
        self.layer4 = self._make_layer(512, 2, stride=2, act_type=act_type)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, act_type: str):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_planes, planes, s, act_type))
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
def get_dataloaders(dataset_name: str = "cifar10", batch_size: int = 128, seed: int = 42):
    dataset_name_lower = str(dataset_name).lower().strip()
    use_pin = torch.cuda.is_available()

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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name_lower == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=use_pin,
        worker_init_fn=seed_worker,
        generator=g
    )
    testloader = DataLoader(
        testset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=use_pin,
        worker_init_fn=seed_worker,
        generator=g
    )
    return trainloader, testloader


# ==========================================
# 4. Training & Benchmark Execution
# ==========================================
def train_single_seed(act_type: str, dataset_name: str = "cifar10", seed: int = 42, epochs: int = 10, device: torch.device = torch.device("cuda")):
    reset_all_seeds(seed)
    trainloader, testloader = get_dataloaders(dataset_name, seed=seed)
    
    model = ResNet18(num_classes=10, act_type=act_type).to(device)
    
    act_params = []
    weight_params = []

    activation_module_types = (AdaptiveAlphaGoLU, AdaptiveSwish, nn.PReLU)
    activation_param_ids = set()
    for module in model.modules():
        if isinstance(module, activation_module_types):
            for parameter in module.parameters(recurse=False):
                if parameter.requires_grad:
                    activation_param_ids.add(id(parameter))
    
    # Explicit parameter separation ensuring zero weight decay on activation parameters
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in activation_param_ids:
            act_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': 1e-3, 'weight_decay': 5e-4},
        {'params': act_params, 'lr': 1e-4, 'weight_decay': 0.0}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    alphas = model.extract_alphas()
    
    return acc, alphas


def run_benchmark(dataset_name: str = "cifar10", seeds: list = [42, 123, 999, 2024, 2025], epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']
    results = {act: [] for act in activations}

    print(f"\n================ Running Unified ResNet-18 Benchmark on {dataset_name.upper()} (N={len(seeds)}) ================")
    
    for act in activations:
        print(f"\n--- Activation: {act.upper()} ---")
        for s in seeds:
            acc, alphas = train_single_seed(act, dataset_name=dataset_name, seed=s, epochs=epochs, device=device)
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


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, dataset_name: str = 'cifar10', epochs: int = 10) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc, _ = train_single_seed(act_type=activation, dataset_name=dataset_name, seed=seed, epochs=epochs, device=device)
    return float(acc)


if __name__ == '__main__':
    default_seeds = [42, 123, 999, 2024, 2025]
    run_benchmark(dataset_name="cifar10", seeds=default_seeds, epochs=10)
    run_benchmark(dataset_name="fashion_mnist", seeds=default_seeds, epochs=10)
