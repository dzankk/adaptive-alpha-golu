"""
Benchmark: Adversarial Robustness on CIFAR-10 (ResNet-18)
Evaluates clean accuracy vs. FGSM and PGD-10 adversarial attack robustness 
across GELU, Swish, Adaptive Swish, Static GoLU, and Adaptive Alpha-GoLU.
Includes proper Gompertz math and CUDA seed resetting.
"""

import inspect
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    class AdaptiveAlphaGoLU(nn.Module):
        """Fallback implementation if local module is missing"""
        def __init__(self, init_alpha=1.0):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        def forward(self, x):
            scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
            return x * torch.exp(-torch.exp(scaled))


def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. Custom Activation Implementations
# ==========================================
class StaticGoLU(nn.Module):
    """Correct & numerically stable Gompertz activation: x * exp(-exp(-x))"""
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class SwishAdaptive(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ==========================================
# 2. Helpers & Factory
# ==========================================
def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        sig = inspect.signature(AdaptiveAlphaGoLU)
        return AdaptiveAlphaGoLU(init_alpha=1.00) if 'init_alpha' in sig.parameters else AdaptiveAlphaGoLU()
    elif act_type == 'swish_adaptive':
        sig = inspect.signature(SwishAdaptive)
        return SwishAdaptive(init_beta=1.00) if 'init_beta' in sig.parameters else SwishAdaptive()
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name:
            act_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{'params': base_params, 'weight_decay': weight_decay}]
    if act_params:
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)


# ==========================================
# 3. ResNet Architecture
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = get_activation(act_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, act_type='relu', num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.act_type = act_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = get_activation(act_type)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.act_type))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out, dim=[2, 3])
        out = self.linear(out)
        return out


# ==========================================
# 4. Adversarial Attack Utilities (Fixed Clamping)
# ==========================================
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def get_normalized_bounds(device):
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    return min_val, max_val


def fgsm_attack(model, images, labels, eps=8/255):
    device = images.device
    min_val, max_val = get_normalized_bounds(device)
    std = CIFAR_STD.to(device)
    
    eps_norm = eps / std
    images_req = images.clone().detach().requires_grad_(True)
    outputs = model(images_req)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbed = images_req + eps_norm * images_req.grad.sign()
    # Element-wise tensor clamping
    return torch.max(torch.min(perturbed, max_val), min_val)


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10):
    device = images.device
    min_val, max_val = get_normalized_bounds(device)
    std = CIFAR_STD.to(device)

    eps_norm = eps / std
    alpha_norm = alpha / std

    ori_images = images.clone().detach()
    perturbed = images.clone().detach() + (torch.rand_like(images) - 0.5) * 2 * eps_norm
    perturbed = torch.max(torch.min(perturbed, max_val), min_val)

    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = perturbed + alpha_norm * perturbed.grad.sign()
        diff = adv_images - ori_images
        
        # Element-wise bounds clamping
        eta = torch.max(torch.min(diff, eps_norm), -eps_norm)
        perturbed = torch.max(torch.min(ori_images + eta, max_val), min_val).detach()

    return perturbed


# ==========================================
# 5. Benchmark Execution
# ==========================================
def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Adversarial Robustness Benchmark on {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    activations = ['relu', 'gelu', 'swish', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        print(f"\n--- Evaluated Activation: {act_type.upper()} ---")
        reset_all_seeds(42)
        model = ResNet18(act_type=act_type).to(device)
        optimizer = get_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        print("Training model for 3 epochs before robustness check...")
        for epoch in range(3):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/3 Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        clean_correct, fgsm_correct, pgd_correct, total = 0, 0, 0, 0

        for idx, (images, labels) in enumerate(test_loader):
            if idx > 15: 
                break  # Fast verification pass across ~1000 images
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                clean_outputs = model(images)
                clean_correct += (clean_outputs.argmax(1) == labels).sum().item()

            adv_fgsm = fgsm_attack(model, images, labels)
            with torch.no_grad():
                fgsm_correct += (model(adv_fgsm).argmax(1) == labels).sum().item()

            adv_pgd = pgd_attack(model, images, labels)
            with torch.no_grad():
                pgd_correct += (model(adv_pgd).argmax(1) == labels).sum().item()

            total += labels.size(0)

        print(f"Clean Accuracy: {clean_correct / total * 100:.2f}%")
        print(f"FGSM Robust Accuracy: {fgsm_correct / total * 100:.2f}%")
        print(f"PGD-10 Robust Accuracy: {pgd_correct / total * 100:.2f}%")


if __name__ == '__main__':
    run_benchmark()
