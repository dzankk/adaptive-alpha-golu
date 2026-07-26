import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from alpha_golu import AlphaGoLU

def main():
    # 1. Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 2. Define standard CNN with AlphaGoLU
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                AlphaGoLU(init_alpha=1.0),
                nn.MaxPool2d(2, 2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                AlphaGoLU(init_alpha=1.0),
                nn.MaxPool2d(2, 2)
            )
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training & Parameter Tracking Loop
    epochs = 5
    print(f"Starting training run on device: {device}\n" + "="*50)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Extract current alpha parameters from Layer 1 and Layer 2
        a1 = model.layer1[1].alpha.item()
        a2 = model.layer2[1].alpha.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Layer 1 Alpha: {a1:.4f} | Layer 2 Alpha: {a2:.4f}")

    print("="*50 + "\nTraining complete!")

if __name__ == '__main__':
    main()
