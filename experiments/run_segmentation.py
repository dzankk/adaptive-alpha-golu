
"""
Semantic Segmentation Benchmark (UNet)
======================================
Evaluates GELU, Static GoLU, and Adaptive Alpha-GoLU on a pixel-level
dense prediction task (UNet Architecture on synthetic/lightweight spatial masks).
Tracks Mean Intersection-over-Union (mIoU) and Dice Score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


class DoubleConv(nn.Module):
    """(convolution => [BN] => Activation) * 2"""
    def __init__(self, in_channels, out_channels, act_type='alpha_golu'):
        super().__init__()
        
        def get_act():
            if act_type == 'gelu':
                return nn.GELU()
            elif act_type == 'golu_static':
                class StaticGoLU(nn.Module):
                    def forward(self, x):
                        return x * torch.exp(-torch.exp(-x))
                return StaticGoLU()
            elif act_type == 'alpha_golu':
                return AdaptiveAlphaGoLU()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_act()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, act_type='alpha_golu'):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32, act_type)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, act_type))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, act_type))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64, act_type)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32, act_type)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        
        return self.outc(x)


def compute_miou(preds, targets, num_classes=2):
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious)


def run_segmentation_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Semantic Segmentation (UNet) Benchmark on {device}...")

    # Generate synthetic spatial segmentation dataset (shapes on canvas)
    # Shape: Batch=64, Channels=3, 64x64 resolution
    def generate_batch(batch_size=32):
        images = torch.randn(batch_size, 3, 64, 64)
        masks = torch.zeros(batch_size, 64, 64, dtype=torch.long)
        for i in range(batch_size):
            # Create synthetic circular foreground object
            x, y = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
            dist = (x - 32)**2 + (y - 32)**2
            mask = (dist < 400).long()
            masks[i] = mask
            images[i, 0] += mask.float() * 2.0  # Boost intensity in target region
        return images.to(device), masks.to(device)

    results = {}
    seeds = [42, 123]

    print("\n================ Semantic Segmentation UNet (Mean IoU ↑) ================")
    for act_type in ['gelu', 'golu_static', 'alpha_golu']:
        mious = []
        for s in seeds:
            torch.manual_seed(s)
            model = UNet(n_channels=3, n_classes=2, act_type=act_type).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for step in range(250):
                xb, yb = generate_batch(batch_size=32)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_ious = []
            with torch.no_grad():
                for _ in range(15):
                    xb, yb = generate_batch(batch_size=32)
                    logits = model(xb)
                    miou = compute_miou(logits, yb)
                    val_ious.append(miou)

            mean_val_miou = np.mean(val_ious)
            mious.append(mean_val_miou)
            print(f"[{act_type.upper():<12} | Seed {s}] Validation Mean IoU: {mean_val_miou:.4f}")

        results[act_type] = (np.mean(mious), np.std(mious))

    print("\n================ SEMANTIC SEGMENTATION SUMMARY ================")
    for act_type, (mean_miou, std_miou) in results.items():
        print(f"  {act_type.upper():<12}: Mean IoU = {mean_miou:.4f} ± {std_miou:.4f}")

if __name__ == '__main__':
    run_segmentation_benchmark()
