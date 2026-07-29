"""
Adversarial Robustness Benchmark (CIFAR-10)
Matches the original GoLU evaluation protocol using FGSM and PGD attacks.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class StaticGoLU(nn.Module):
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))


def get_activation(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'swish_adaptive':
        return AdaptiveSwish(init_beta=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=0.50) if hasattr(AdaptiveAlphaGoLU, '__init__') and 'init_alpha' in AdaptiveAlphaGoLU.__init__.__code__.co_varnames else AdaptiveAlphaGoLU()
    raise ValueError(f"Unknown activation: {act_type}")


class ResBlock(nn.Module):
    def __init__(self, channels, act_type):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = get_activation(act_type)

    def forward(self, x):
        return self.act2(x + self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x))))))


class SmallCIFARNet(nn.Module):
    def __init__(self, act_type='alpha_golu'):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            get_activation(act_type)
        )
        self.block1 = ResBlock(32, act_type)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    return torch.clamp(images + epsilon * images.grad.sign(), -1, 1)


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=5):
    orig = images.clone().detach()
    adv = images.clone().detach()

    for _ in range(iters):
        adv.requires_grad = True
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        step = adv + alpha * adv.grad.sign()
        eta = torch.clamp(step - orig, min=-epsilon, max=epsilon)
        adv = torch.clamp(orig + eta, min=-1, max=1).detach()

    return adv


def run_robustness_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Real CIFAR-10 Adversarial Robustness Benchmark on {device}...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    epsilon = 0.03

    print(f"\n================ CIFAR-10 Adversarial Robustness (Epsilon = {epsilon}) ================")
    for act in activations:
        reset_seeds(42)
        model = SmallCIFARNet(act_type=act).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(3):
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        model.eval()
        clean_accs, fgsm_accs, pgd_accs = [], [], []

        for i, (x, y) in enumerate(testloader):
            if i >= 10:
                break
            x, y = x.to(device), y.to(device)
            clean_accs.append((model(x).argmax(dim=1) == y).float().mean().item())

            x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
            fgsm_accs.append((model(x_fgsm).argmax(dim=1) == y).float().mean().item())

            x_pgd = pgd_attack(model, x, y, epsilon=epsilon)
            pgd_accs.append((model(x_pgd).argmax(dim=1) == y).float().mean().item())

        print(f"[{act.upper():<14}] Clean Acc: {np.mean(clean_accs)*100:.2f}% | FGSM Acc: {np.mean(fgsm_accs)*100:.2f}% | PGD Acc: {np.mean(pgd_accs)*100:.2f}%")


if __name__ == '__main__':
    run_robustness_benchmark()
