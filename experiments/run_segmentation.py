"""
Benchmark: Semantic Segmentation (U-Net)
Measures pixel-level target segmentation performance (mIoU) across activation functions.
Demonstrates layer skip-connections combined with parameter-group optimization 
(disabling weight decay for trainable activation variables like alpha and beta).
"""
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Custom Activations
# ==========================================
class GoLUStatic(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class AdaptiveAlphaGoLU(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return x * torch.sigmoid(1.702 * self.alpha * x)

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
    elif act_type == 'golu_static':
        return GoLUStatic()
    elif act_type == 'alpha_golu':
        sig = inspect.signature(AdaptiveAlphaGoLU)
        return AdaptiveAlphaGoLU(init_alpha=0.50) if 'init_alpha' in sig.parameters else AdaptiveAlphaGoLU()
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
        param_groups.append({'params': act_params, 'lr': lr * 5.0, 'weight_decay': 0.0})

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

def run_segmentation_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Segmentation Benchmark on {device}")

    loader = DataLoader(SyntheticSegmentationDataset(), batch_size=8, shuffle=True)
    activations = ['relu', 'gelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        model = UNet(act_type=act_type).to(device)
        optimizer = get_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(2):
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

        mean_iou = total_iou / len(loader)
        print(f"Activation: {act_type.ljust(15)} | Validation mIoU: {mean_iou:.4f}")

if __name__ == '__main__':
    run_segmentation_benchmark()
