"""
ResNet-18 Multi-Seed Benchmark (Optimized Training + P-GELU)
=============================================================
Evaluates GELU, Static GoLU, Parametric GELU (P-GELU), and Adaptive Alpha-GoLU
on ResNet-18 with Cosine Annealing learning rate schedule across N=3 seeds.
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from models.baselines import ParametricGELU

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    try:
        from models.alpha_golu import AdaptiveGoLU as AdaptiveAlphaGoLU
    except ImportError:
        from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


def replace_relus(model, act_type):
    """Replaces ReLUs in ResNet modules with the target activation function."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            if act_type == 'gelu':
                setattr(model, name, nn.GELU())
            elif act_type == 'golu_static':
                class StaticGoLU(nn.Module):
                    def forward(self, x):
                        return x * torch.exp(-torch.exp(-x))
                setattr(model, name, StaticGoLU())
            elif act_type == 'pgelu':
                setattr(model, name, ParametricGELU(init_alpha=1.0))
            elif act_type == 'alpha_golu':
                setattr(model, name, AdaptiveAlphaGoLU())
        else:
            replace_relus(module, act_type)


def run_resnet_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    seeds = [42, 123, 999]
    results = {}
    epochs = 10

    print("================ ResNet-18 Multi-Seed Benchmark (10 Epochs + Cosine LR) ================")
    for act_type in ['gelu', 'golu_static', 'pgelu', 'alpha_golu']:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
            replace_relus(model, act_type)
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
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
            accs.append(acc)
            print(f"[{act_type.upper():<12} | Seed {s}] ResNet-18 Test Acc: {acc:.2f}%")

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[act_type] = (mean_acc, std_acc)

    print("\n================ RESNET-18 OPTIMIZED SUMMARY ================")
    for act_type, (mean, std) in results.items():
        print(f"  {act_type.upper():<12}: {mean:.2f}% ± {std:.2f}%")


if __name__ == '__main__':
    run_resnet_eval()
