"""
MultiDepth Vision Benchmark
============================
Trains Alpha-GoLU, extracts layerwise alpha trajectories, and renders final charts.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from models.baselines import DeepConvNet
from utils.visualizer import plot_experiment_dashboard

def run_vision_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    model = DeepConvNet(act_type='alpha_golu', num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    results = {'alpha_golu': {'train_loss': [], 'val_acc': [], 'alpha_history': [], 'latent_var': []}}

    print("Running Multi-depth Vision Training Loop...")
    for epoch in range(5):
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

        # Validation
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
        avg_loss = running_loss / len(trainloader)

        results['alpha_golu']['train_loss'].append(avg_loss)
        results['alpha_golu']['val_acc'].append(acc)
        results['alpha_golu']['alpha_history'].append(model.extract_alphas())
        results['alpha_golu']['latent_var'].append(model.extract_latent_variances())

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Alphas: {model.extract_alphas()}")

    plot_experiment_dashboard(results, save_dir="outputs")

if __name__ == '__main__':
    run_vision_benchmark()
