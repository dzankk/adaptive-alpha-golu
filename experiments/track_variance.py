"""
Empirical Latent Space Variance Tracker
=======================================
measures feature variance squeezing across network depth
"""

import torch
import torchvision
import torchvision.transforms as transforms
from models.baselines import DeepConvNet

def track_latent_variance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    model = DeepConvNet(act_type='alpha_golu', num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("\nTracking Latent Layer Variances over 3 Epochs...")
    for epoch in range(3):
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        vars_logged = model.extract_latent_variances()
        print(f"Epoch [{epoch+1}/3] | Layer Variances (L1->L4): {[round(v, 4) for v in vars_logged]}")

if __name__ == '__main__':
    track_latent_variance()
