"""
Benchmark: Adversarial Robustness on CIFAR-10 (ResNet-18)
=========================================================
Evaluates clean accuracy vs. PGD-10 adversarial attack robustness 
across ReLU, GELU, Swish, PReLU, PGELU, Static GoLU, Adaptive Alpha-GoLU, and Adaptive Swish.
Includes proper Gompertz math, deterministic PGD evaluation, and CUDA seed resetting.
"""

import math
import inspect
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
except ImportError:
    class StaticGoLU(nn.Module):
        """Correct & numerically stable Gompertz activation: x * exp(-exp(-x))"""
        def forward(self, x):
            scaled = torch.clamp(-x, min=-88.0, max=88.0)
            return x * torch.exp(-torch.exp(scaled))

    class AdaptiveAlphaGoLU(nn.Module):
        """Fallback implementation using Softplus for alpha positivity"""
        def __init__(self, init_alpha=1.0):
            super().__init__()
            init_val = float(init_alpha)
            init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
            self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

        @property
        def alpha(self):
            return nn.functional.softplus(self.raw_alpha)

        def forward(self, x):
            scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
            return x * torch.exp(-torch.exp(scaled))


def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_param_ids = set()
    
    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, StaticGoLU, PGELU, SwishAdaptive, AdaptiveSwish, nn.PReLU)):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    act_param_ids.add(id(p))

    act_params = []
    base_params = []

    for p in model.parameters():
        if p.requires_grad:
            if id(p) in act_param_ids:
                act_params.append(p)
            else:
                base_params.append(p)

    param_groups = [{'params': base_params, 'weight_decay': weight_decay}]
    if act_params:
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)


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
# 4. Adversarial Attack Utilities
# ==========================================
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def get_normalized_bounds(device):
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    return min_val, max_val


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10):
    """Standard PGD-10 Attack with strictly bounded random initialization."""
    device = images.device
    min_val, max_val = get_normalized_bounds(device)
    std = CIFAR_STD.to(device)

    eps_norm = eps / std
    alpha_norm = alpha / std

    ori_images = images.clone().detach()
    
    # Standard random initialization within epsilon ball & valid image space
    random_noise = torch.empty_like(images).uniform_(-1.0, 1.0) * eps_norm
    perturbed = torch.clamp(ori_images + random_noise, ori_images - eps_norm, ori_images + eps_norm)
    perturbed = torch.clamp(perturbed, min_val, max_val).detach()

    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()

        if perturbed.grad is None:
            break

        adv_images = perturbed + alpha_norm * perturbed.grad.sign()
        diff = adv_images - ori_images
        
        eta = torch.clamp(diff, -eps_norm, eps_norm)
        perturbed = torch.clamp(ori_images + eta, min_val, max_val).detach()

    return perturbed


# ==========================================
# 5. Benchmark Execution Functions
# ==========================================
def train_single_seed_robustness(act_type: str, seed: int, epochs: int, device: torch.device) -> tuple[float, float]:
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_g,
    )
    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_g,
    )

    model = ResNet18(act_type=act_type).to(device)
    optimizer = get_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    clean_correct, pgd_correct, total = 0, 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_correct += (model(images).argmax(1) == labels).sum().item()

        with torch.enable_grad():
            adv_pgd = pgd_attack(model, images, labels)

        with torch.no_grad():
            pgd_correct += (model(adv_pgd).argmax(1) == labels).sum().item()

        total += labels.size(0)

    clean_acc = (clean_correct / total) * 100.0 if total > 0 else 0.0
    pgd_acc = (pgd_correct / total) * 100.0 if total > 0 else 0.0
    return clean_acc, pgd_acc


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Full Adversarial Robustness Benchmark on {device}")
    activations = ['relu', 'gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        clean_acc, pgd_acc = train_single_seed_robustness(act_type=act_type, seed=42, epochs=5, device=device)
        print(f"Activation: {act_type.ljust(15)} | Clean Acc: {clean_acc:.2f}% | PGD-10 Robust Acc: {pgd_acc:.2f}%")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Robust Accuracy under PGD attack."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, robust_acc = train_single_seed_robustness(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(robust_acc)


if __name__ == '__main__':
    run_benchmark()
