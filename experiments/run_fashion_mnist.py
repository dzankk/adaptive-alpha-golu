"""
Fashion-MNIST Cross-Dataset Generalization Benchmark
=====================================================
Evaluates model adaptability on grayscale structured item patterns.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from models.baselines import DeepConvNet


def run_fashion_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(3),  # Expand 1-channel grayscale to 3 channels for DeepConvNet
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    print("================ Fashion-MNIST Generalization Benchmark ================")
    for act_type in ['gelu', 'golu_static', 'alpha_golu']:
        torch.manual_seed(42)
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

        print(f"[{act_type.upper()}] Fashion-MNIST Test Accuracy: {100.0 * correct / total:.2f}%")


if __name__ == '__main__':
    run_fashion_eval()
