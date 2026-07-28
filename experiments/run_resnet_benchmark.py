
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from models.alpha_golu import AdaptiveAlphaGoLU

def replace_relus_with_alpha_golu(model):
    """Recursively replaces standard ReLU activations with AdaptiveAlphaGoLU."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, AdaptiveAlphaGoLU())
        else:
            replace_relus_with_alpha_golu(module)

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

    print("================ Evaluating ResNet-18 with Alpha-GoLU ================")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Swap out standard ReLU with our Adaptive Alpha-GoLU
    replace_relus_with_alpha_golu(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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
        print(f"Epoch [{epoch+1}/5] | Loss: {running_loss/len(trainloader):.4f} | ResNet-18 Test Acc: {acc:.2f}%")

if __name__ == '__main__':
    run_resnet_eval()
