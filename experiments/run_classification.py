"""
Benchmark: Consolidated Image Classification & Trajectory Analysis
Unified ResNet-18 runner for CIFAR-10 and Fashion-MNIST across multi-seed evaluations.
Supports ReLU, GELU, Swish, Adaptive Swish, PReLU, PGELU, Static GoLU, and Adaptive Alpha-GoLU.
Includes strict softplus alpha constraints, robust statistics, and memory-decoupled architecture.
"""

import inspect
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fallback robust statistical calculation using SciPy/NumPy
try:
    from utils.metrics import compute_summary_statistics, calculate_p_value
except ImportError:
    try:
        from utils.stats import compute_summary_statistics, calculate_p_value
    except ImportError:
        def compute_summary_statistics(data):
            arr = np.array(data, dtype=np.float64)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            }

        def calculate_p_value(a, b):
            """Welch's t-test for unequal variances."""
            try:
                from scipy import stats
                _, p_val = stats.ttest_ind(a, b, equal_var=False)
                return float(p_val)
            except Exception:
                # Basic Gaussian approximation fallback
                a_arr, b_arr = np.array(a), np.array(b)
                mean_a, mean_b = np.mean(a_arr), np.mean(b_arr)
                var_a = np.var(a_arr, ddof=1) / len(a_arr)
                var_b = np.var(b_arr, ddof=1) / len(b_arr)
                se = np.sqrt(var_a + var_b)
                if se == 0:
                    return 1.0
                z = abs(mean_a - mean_b) / se
                # Standard normal approximation for p-value
                p_val = float(2.0 * (1.0 - 0.5 * (1.0 + torch.erf(torch.tensor(z / np.sqrt(2.0))).item())))
                return p_val


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
class StaticGoLU(nn.Module):
    """Numerically stable Gompertz Linear Unit: x * exp(-exp(-x))"""
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class AdaptiveAlphaGoLU(nn.Module):
    """Adaptive Gompertz Linear Unit with softplus constrained positive alpha."""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        # Softplus inverse initialization: softplus(init_raw) = init_alpha
        init_raw = np.log(np.exp(init_val) - 1.0) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        """Ensures alpha stays strictly positive (> 0) during gradient descent."""
        return F.softplus(self.raw_alpha)

    def forward(self, x):
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x) with softplus constrained positive alpha."""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = np.log(np.exp(init_val) - 1.0) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    """Adaptive Swish: x * sigmoid(beta * x)"""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    """Activation factory function."""
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
    def __init__(self, in_planes, planes, stride=1, act_type='alpha_golu'):
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

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, act_type='alpha_golu'):
        super().__init__()
        self.in_planes = 64
        self.act_type = act_type
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = get_activation(act_type)

        self.layer1 = self._make_layer(64, 2, stride=1, act_type=act_type)
        self.layer2 = self._make_layer(128, 2, stride=2, act_type=act_type)
        self.layer3 = self._make_layer(256, 2, stride=2, act_type=act_type)
        self.layer4 = self._make_layer(512, 2, stride=2, act_type=act_type)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, act_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_planes, planes, s, act_type))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out, dim=[2, 3])  # Spatial Adaptive Pooling (B, 512)
        out = self.linear(out)
        return out

    def extract_alphas(self):
        alphas = []
        for module in self.modules():
            if isinstance(module, (AdaptiveAlphaGoLU, AdaptiveSwish, PGELU)):
                # Return evaluated softplus alpha property values
                alphas.append(module.alpha.detach().cpu().item())
            elif isinstance(module, nn.PReLU):
                alphas.extend(module.weight.detach().cpu().numpy().flatten())
        return alphas


def get_dataloaders(dataset_name="cifar10", batch_size=128):
    dataset_name_lower = str(dataset_name).lower().strip()
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def train_single_seed(act_type, dataset_name="cifar10", seed=42, epochs=10, device="cuda"):
    reset_all_seeds(seed)
    trainloader, testloader = get_dataloaders(dataset_name)
    
    model = ResNet18(num_classes=10, act_type=act_type).to(device)
    
    act_params = []
    weight_params = []
    
    # Explicit memory ID-based parameter isolation
    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, AdaptiveSwish, PGELU, nn.PReLU)):
            for p in module.parameters():
                if p.requires_grad:
                    act_params.append(p)

    act_params_ids = set(map(id, act_params))
    for p in model.parameters():
        if p.requires_grad and id(p) not in act_params_ids:
            weight_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': 1e-3, 'weight_decay': 5e-4},
        {'params': act_params, 'lr': 1e-4, 'weight_decay': 0.0}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        
        # Warmup: freeze activation parameters during first 2 epochs for stability
        if act_params:
            requires_grad = (epoch >= 2)
            for p in act_params:
                p.requires_grad = requires_grad

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


def run_benchmark(dataset_name="cifar10", seeds=[42, 123, 999, 2024, 2025], epochs=10):
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
                print(f"[{act.upper():<14} | Seed {s}] Accuracy: {acc:.2f}% | Final Mean Alpha: {mean_alpha:.4f}")
            else:
                print(f"[{act.upper():<14} | Seed {s}] Accuracy: {acc:.2f}%")

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
