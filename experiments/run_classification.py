"""
Consolidated Classification & Trajectory Benchmark
===================================================
Unified ResNet-18 runner for CIFAR-10 and Fashion-MNIST across multi-seed evaluations.
Supports GELU, Swish, Adaptive Swish, PReLU, Static GoLU, and Adaptive Alpha-GoLU.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.metrics import compute_summary_statistics, calculate_p_value

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


def reset_all_seeds(seed=42):
    """Ensures true reproducibility across CUDA operations and activations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class StaticGoLU(nn.Module):
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))


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

        self.act_type = act_type
        self.act1 = self._get_act()
        self.act2 = self._get_act()

    def _get_act(self):
        if self.act_type == 'gelu':
            return nn.GELU()
        elif self.act_type == 'swish':
            return nn.SiLU()
        elif self.act_type == 'swish_adaptive':
            return AdaptiveSwish()
        elif self.act_type == 'prelu':
            return nn.PReLU()
        elif self.act_type == 'golu_static':
            return StaticGoLU()
        elif self.act_type == 'alpha_golu':
            return AdaptiveAlphaGoLU()
        else:
            raise ValueError(f"Unknown activation type: {self.act_type}")

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.act_type = act_type
        if act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'swish':
            self.act = nn.SiLU()
        elif act_type == 'swish_adaptive':
            self.act = AdaptiveSwish()
        elif act_type == 'prelu':
            self.act = nn.PReLU()
        elif act_type == 'golu_static':
            self.act = StaticGoLU()
        elif act_type == 'alpha_golu':
            self.act = AdaptiveAlphaGoLU()

        self.layer1 = self._make_layer(64, 2, stride=1, act_type=act_type)
        self.layer2 = self._make_layer(128, 2, stride=2, act_type=act_type)
        self.layer3 = self._make_layer(256, 2, stride=2, act_type=act_type)
        self.layer4 = self._make_layer(512, 2, stride=2, act_type=act_type)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, act_type):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def extract_alphas(self):
        alphas = []
        for m in self.modules():
            if isinstance(m, AdaptiveAlphaGoLU):
                alphas.extend(m.alpha.detach().cpu().numpy().flatten())
        return alphas


def get_dataloaders(dataset_name="cifar10", batch_size=128):
    if dataset_name.lower() == "cifar10":
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
    elif dataset_name.lower() == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    return trainloader, testloader


def train_single_seed(act_type, dataset_name="cifar10", seed=42, epochs=10, device="cuda"):
    reset_all_seeds(seed)
    trainloader, testloader = get_dataloaders(dataset_name)
    
    model = ResNet18(num_classes=10, act_type=act_type).to(device)
    
    # Decouple parameter learning rate for alpha parameters
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n or 'beta' in n]
    weight_params = [p for n, p in model.named_parameters() if 'alpha' not in n and 'beta' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': 1e-3, 'weight_decay': 5e-4},
        {'params': alpha_params, 'lr': 5e-4, 'weight_decay': 0.0}
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

    # Evaluation
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


def run_benchmark(dataset_name="cifar10", seeds=[42, 123, 999], epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = ['gelu', 'swish', 'swish_adaptive', 'prelu', 'golu_static', 'alpha_golu']
    results = {act: [] for act in activations}

    print(f"\n================ Running Unified ResNet-18 Benchmark on {dataset_name.upper()} (N={len(seeds)}) ================")
    
    for act in activations:
        print(f"\n--- Activation: {act.upper()} ---")
        for s in seeds:
            acc, alphas = train_single_seed(act, dataset_name=dataset_name, seed=s, epochs=epochs, device=device)
            results[act].append(acc)
            if act == 'alpha_golu' and alphas:
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


if __name__ == '__main__':
    run_benchmark(dataset_name="cifar10", seeds=[42, 123, 999], epochs=10)
    run_benchmark(dataset_name="fashion_mnist", seeds=[42, 123, 999], epochs=10)
