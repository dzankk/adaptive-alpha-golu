"""
Benchmark: Semantic Segmentation (U-Net)
Measures pixel-level target segmentation performance (mIoU) across activation functions.
Demonstrates layer skip-connections combined with parameter-group optimization 
(disabling weight decay for trainable activation variables like alpha and beta).
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU


# ==========================================
# 1. Custom Activation Implementations
# ==========================================
class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x)"""
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    """Adaptive Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type in ('adaptive_swish', 'swish_adaptive'):
        return SwishAdaptive(init_beta=1.0)
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    
    for module in model.modules():
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
        # Zero weight-decay applied strictly to trainable activation parameters
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)


# ==========================================
# 2. U-Net Architecture
# ==========================================
class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_type: str):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(act_type),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, act_type: str = 'relu'):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Generates structured synthetic geometric targets for segmentation learning."""
    def __init__(self, size: int = 100, height: int = 128, width: int = 128, seed: int = 42):
        self.size = size
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        sample_rng = np.random.RandomState(self.seed + idx)
        
        # Base structured noise background
        x = sample_rng.randn(3, self.height, self.width).astype(np.float32)
        
        # Draw dynamic synthetic target region
        y = np.zeros((1, self.height, self.width), dtype=np.float32)
        cx, cy = sample_rng.randint(32, 96, size=2)
        r = sample_rng.randint(16, 32)
        
        grid_y, grid_x = np.ogrid[:self.height, :self.width]
        mask = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= r ** 2
        
        y[0, mask] = 1.0
        x[0, mask] += 1.5  # Signal shift inside target mask
        
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_mIoU(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_seed_segmentation(act_type: str, seed: int, epochs: int, device: torch.device) -> float:
    set_seed(seed)
    
    full_dataset = SyntheticSegmentationDataset(size=100, seed=seed)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=loader_g)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, generator=loader_g)

    model = UNet(act_type=act_type).to(device)
    optimizer = get_optimizer(model, lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_iou = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_iou += compute_mIoU(out, y)

    return total_iou / len(val_loader)


def run_segmentation_benchmark(seeds=[42, 123, 999, 2024, 2025], epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Segmentation Benchmark on {device} (N={len(seeds)})")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']

    for act_type in activations:
        scores = []
        for s in seeds:
            miou = train_single_seed_segmentation(act_type=act_type, seed=s, epochs=epochs, device=device)
            scores.append(miou)
            print(f"Activation: {act_type.ljust(15)} | Seed {s} | Validation mIoU: {miou:.4f}")
        print(f"--> {act_type.upper()} Mean mIoU: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Mean Intersection over Union (mIoU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miou = train_single_seed_segmentation(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(miou)


if __name__ == '__main__':
    run_segmentation_benchmark(seeds=[42, 123, 999, 2024, 2025], epochs=10)
