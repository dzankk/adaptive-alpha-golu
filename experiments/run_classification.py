"""
Consolidated Classification & Layerwise Trajectory Benchmark
=============================================================
Unified runner for CIFAR-10 and Fashion-MNIST across multi-seed evaluations.
Logs layerwise alpha trajectories, latent variance squeezing, and p-values.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.baselines import DeepConvNet
from utils.metrics import compute_summary_statistics, calculate_p_value


def get_dataloaders(dataset_name="cifar10", batch_size=128):
    if dataset_name.lower() == "cifar10":
        transform_train = transforms.Compose([
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


def train_single_seed(act_type, dataset_name="cifar10", seed=42, epochs=5, device="cuda"):
    torch.manual_seed(seed)
    trainloader, testloader = get_dataloaders(dataset_name)
    
    model = DeepConvNet(act_type=act_type, num_classes=10).to(device)
    
    # Decouple parameter learning rate for alpha parameters
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n or 'beta' in n]
    weight_params = [p for n, p in model.named_parameters() if 'alpha' not in n and 'beta' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': weight_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': alpha_params, 'lr': 5e-4, 'weight_decay': 0.0}
    ])
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
    
    # Extract structural metrics if available
    alphas = model.extract_alphas() if hasattr(model, 'extract_alphas') else []
    variances = model.extract_latent_variances() if hasattr(model, 'extract_latent_variances') else []
    
    return acc, alphas, variances


def run_benchmark(dataset_name="cifar10", seeds=[42, 123, 999], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = ['gelu', 'swish', 'prelu', 'golu_static', 'alpha_golu']
    results = {act: [] for act in activations}

    print(f"\n================ Running Unified Benchmark on {dataset_name.upper()} (N={len(seeds)}) ================")
    
    for act in activations:
        print(f"\n--- Activation: {act.upper()} ---")
        for s in seeds:
            acc, alphas, vars_logged = train_single_seed(act, dataset_name=dataset_name, seed=s, epochs=epochs, device=device)
            results[act].append(acc)
            print(f"[{act.upper():<12} | Seed {s}] Accuracy: {acc:.2f}%")
            if act == 'alpha_golu' and alphas:
                print(f"   ↳ Final Layer Alphas: {[round(a, 4) for a in alphas]}")

    print(f"\n================ {dataset_name.upper()} SUMMARY STATISTICS ================")
    for act, accs in results.items():
        stats_res = compute_summary_statistics(accs)
        print(f"  {act.upper():<12}: Mean = {stats_res['mean']:.2f}% ± {stats_res['std']:.2f}%")

    if 'golu_static' in results and 'alpha_golu' in results:
        p_val = calculate_p_value(results['golu_static'], results['alpha_golu'])
        print(f"\nStatistical Significance (Alpha-GoLU vs Static GoLU p-value): {p_val:.4f}")


if __name__ == '__main__':
    run_benchmark(dataset_name="cifar10", seeds=[42, 123, 999], epochs=5)
    run_benchmark(dataset_name="fashion_mnist", seeds=[42, 123, 999], epochs=5)
