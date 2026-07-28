"""
Multi-Seed Comparative Benchmark Runner
========================================
Executes model training across multiple random seeds to gather statistical metrics.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from models.baselines import DeepConvNet
from utils.metrics import compute_summary_statistics, calculate_p_value

def run_multi_seed_benchmark(seeds=[42, 123, 999], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    activations = ['gelu', 'golu_static', 'alpha_golu']
    final_results = {act: [] for act in activations}

    for act in activations:
        print(f"\n================ Evaluating {act.upper()} ================")
        for seed in seeds:
            torch.manual_seed(seed)
            model = DeepConvNet(act_type=act, num_classes=10).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(epochs):
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Eval
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
            final_results[act].append(acc)
            print(f"Seed {seed} -> Accuracy: {acc:.2f}%")

    print("\n================ STATISTICAL SUMMARY ================")
    for act, accs in final_results.items():
        stats_res = compute_summary_statistics(accs)
        print(f"{act.upper()}: Mean Acc = {stats_res['mean']:.2f}% ± {stats_res['std']:.2f}%")

    p_val = calculate_p_value(final_results['golu_static'], final_results['alpha_golu'])
    print(f"\nStatistical Significance (p-value alpha-GoLU vs Static GoLU): {p_val:.4f}")

if __name__ == '__main__':
    run_multi_seed_benchmark()
