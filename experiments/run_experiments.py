"""
Main Experimental Pipeline & Benchmark Suite
=============================================
Runs multi-baseline comparative evaluations (GELU, Swish, Static GoLU, Alpha-GoLU)
under fixed seed constraints. Extracts metric histories and invokes visualizer routines.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.baselines import DeepConvNet
from utils.visualizer import plot_experiment_dashboard


def set_deterministic_seed(seed: int = 42):
    """Ensures absolute reproducibility across experimental runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_single_model(act_type: str, trainloader, testloader, epochs: int = 5, device: str = 'cpu'):
    """Trains a given activation variant and extracts metrics."""
    print(f"\n---> Commencing Training Run: Activation = [{act_type.upper()}]")
    model = DeepConvNet(act_type=act_type, num_classes=10, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    val_acc_history = []
    alpha_history = []
    latent_var_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_loss_history.append(epoch_loss)

        # Validation Phase
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
        val_acc_history.append(acc)

        # Log internal dynamics
        current_alphas = model.extract_alphas()
        current_vars = model.extract_latent_variances()
        alpha_history.append(current_alphas)
        latent_var_history.append(current_vars)

        if act_type == 'alpha_golu':
            alpha_str = " | Alphas: " + ", ".join([f"{a:.3f}" for a in current_alphas])
        else:
            alpha_str = ""

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Val Acc: {acc:.2f}%{alpha_str}")

    return {
        'train_loss': train_loss_history,
        'val_acc': val_acc_history,
        'alpha_history': alpha_history,
        'latent_var': latent_var_history
    }


def main():
    set_deterministic_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Adaptive Alpha-GoLU Research Suite ===")
    print(f"Device Allocated: {device}\n")

    # Data Pipeline (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("Loading CIFAR-10 benchmark dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    baselines = ['gelu', 'swish', 'golu_static', 'alpha_golu']
    results = {}
    epochs_per_run = 5  # Scalable for longer training runs

    for act in baselines:
        results[act] = train_single_model(
            act_type=act,
            trainloader=trainloader,
            testloader=testloader,
            epochs=epochs_per_run,
            device=device
        )

    # Render visualizations
    plot_experiment_dashboard(results, save_dir="outputs")
    print("\n=== Benchmark Complete! Results and dashboard saved to /outputs ===")


if __name__ == '__main__':
    main()
