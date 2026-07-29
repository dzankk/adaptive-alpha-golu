"""
ResNet-18 Benchmark with GELU, Swish, Adaptive Swish, Static GoLU, and Adaptive Alpha-GoLU
Includes deterministic seed resetting and trajectory tracking for learnable parameters.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        elif self.act_type == 'golu_static':
            return StaticGoLU()
        elif self.act_type == 'alpha_golu':
            return AdaptiveAlphaGoLU()

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
        
        if act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'swish':
            self.act = nn.SiLU()
        elif act_type == 'swish_adaptive':
            self.act = AdaptiveSwish()
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

    def get_alpha_values(self):
        alphas = []
        for m in self.modules():
            if isinstance(m, AdaptiveAlphaGoLU):
                alphas.extend(m.alpha.detach().cpu().numpy().flatten())
        return alphas


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ResNet-18 Multi-Seed Benchmark on {device}...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testset_transform if 'testset_transform' in locals() else transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    seeds = [42, 123, 999]
    results = {}

    print("\n================ Comprehensive ResNet-18 Benchmark (10 Epochs) ================")
    for act_type in activations:
        accs = []
        for s in seeds:
            reset_all_seeds(s)  # <-- Fix A: Reset CUDA RNG state on every single iteration
            model = ResNet18(num_classes=10, act_type=act_type).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=10)

            for epoch in range(10):
                model.train()
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100.0 * correct / total
            accs.append(acc)
            
            if act_type == 'alpha_golu':
                alphas = model.get_alpha_values()
                mean_alpha = np.mean(alphas) if len(alphas) > 0 else 1.0
                print(f"[{act_type.upper():<14} | Seed {s}] Accuracy: {acc:.2f}% | Final Mean Alpha: {mean_alpha:.4f}")
            else:
                print(f"[{act_type.upper():<14} | Seed {s}] Accuracy: {acc:.2f}%")

        results[act_type] = (np.mean(accs), np.std(accs))

    print("\n================ RESNET-18 BENCHMARK SUMMARY ================")
    for act_type, (mean_acc, std_acc) in results.items():
        print(f"  {act_type.upper():<14}: {mean_acc:.2f}% ± {std_acc:.2f}%")

if __name__ == '__main__':
    run_benchmark()
