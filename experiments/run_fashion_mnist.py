
"""
Fashion-MNIST Cross-Dataset Generalization Benchmark
=====================================================
Multi-seed (N=3) evaluation of model adaptability on grayscale items.
"""

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from models.baselines import DeepConvNet


def run_fashion_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    seeds = [42, 123, 999]
    results = {}

    print("================ Fashion-MNIST Multi-Seed Benchmark (N=3) ================")
    for act_type in ['gelu', 'golu_static', 'alpha_golu']:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            model = DeepConvNet(act_type=act_type, num_classes=10).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(5):
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()

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
            print(f"[{act_type.upper()} | Seed {s}] Fashion-MNIST Acc: {acc:.2f}%")

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[act_type] = (mean_acc, std_acc)

    print("\n================ FASHION-MNIST RESULTS SUMMARY ================")
    for act_type, (mean, std) in results.items():
        print(f"  {act_type.upper():<12}: {mean:.2f}% ± {std:.2f}%")


if __name__ == '__main__':
    run_fashion_eval()
