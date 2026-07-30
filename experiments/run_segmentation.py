"""
Benchmark: Semantic Segmentation (U-Net)
Measures pixel-level target segmentation performance (mIoU) across activation functions.
Demonstrates layer skip-connections combined with parameter-group optimization 
(disabling weight decay for trainable activation variables like alpha and beta).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Fixed Custom Activations
# ==========================================
class GoLUStatic(nn.Module):
    """Gompertz Linear Unit: x * exp(-exp(-x))"""
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class AdaptiveAlphaGoLU(nn.Module):
    """Adaptive Gompertz Linear Unit: x * exp(-exp(-alpha * x))"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x)"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type == 'golu_static':
        return GoLUStatic()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    elif act_type == 'swish_adaptive':
        return SwishAdaptive(init_beta=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    
    for module_name, module in model.named_modules():
        if isinstance(module, (AdaptiveAlphaGoLU, PGELU, SwishAdaptive, nn.PReLU)):
            for p in module.parameters():
                if p.requires_grad:
                    act_params.append(p)

    act_param_ids = set(map(id, act_params))
    for p in model.parameters():
        if p.requires_grad and id(p) not in act_param_ids:
            base_params.append(p)

    param_groups = [{'params': base_params, 'weight_decay': weight_decay}]
    if act_params:
        # Zero weight-decay applied to trainable activation variables
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)

# ==========================================
# 2. U-Net Architecture
# ==========================================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act_type):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            get_activation(act_type),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            get_activation(act_type)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, act_type='relu'):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 32, act_type)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(32, 64, act_type)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = UNetBlock(64, 128, act_type)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64, act_type)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32, act_type)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

# ==========================================
# 3. Synthetic Dataset & Metrics
# ==========================================
class SyntheticSegmentationDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = torch.randn(3, 128, 128)
        y = (x[0:1] > 0.5).float()
        return x, y


def compute_mIoU(preds, targets, threshold=0.5):
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_binary * targets).sum(dim=[2, 3])
    union = (preds_binary + targets - preds_binary * targets).sum(dim=[2, 3])
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_seed_segmentation(act_type: str, seed: int, epochs: int, device: torch.device) -> float:
    set_seed(seed)
    loader = DataLoader(SyntheticSegmentationDataset(), batch_size=8, shuffle=True)
    model = UNet(act_type=act_type).to(device)
    optimizer = get_optimizer(model)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_iou = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_iou += compute_mIoU(out, y)

    return total_iou / len(loader)


def run_segmentation_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Segmentation Benchmark on {device}")
    activations = ['gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']

    for act_type in activations:
        mean_iou = train_single_seed_segmentation(act_type=act_type, seed=42, epochs=2, device=device)
        print(f"Activation: {act_type.ljust(15)} | Validation mIoU: {mean_iou:.4f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Mean Intersection over Union (mIoU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miou = train_single_seed_segmentation(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(miou)


if __name__ == '__main__':
    run_segmentation_benchmark()
