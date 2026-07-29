
"""
Semantic Segmentation Benchmark (UNet with Noisy Textured Target Masks)
Prevents trivial 1.0000 mIoU outputs by using challenging target boundaries.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class StaticGoLU(nn.Module):
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='alpha_golu'):
        super().__init__()
        def get_act():
            if act_type == 'gelu':
                return nn.GELU()
            elif act_type == 'swish':
                return nn.SiLU()
            elif act_type == 'swish_adaptive':
                return AdaptiveSwish()
            elif act_type == 'golu_static':
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
        self.conv_up2 = DoubleConv(32*2, 32, act_type)
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
    return np.mean(ious) if ious else 0.0


def generate_hard_batch(batch_size=32, device='cuda'):
    # Creates noisy textured spatial masks (stochastic shapes + high noise background)
    images = torch.randn(batch_size, 3, 64, 64) * 1.5
    masks = torch.zeros(batch_size, 64, 64, dtype=torch.long)
    for i in range(batch_size):
        cx, cy = np.random.randint(20, 44, size=2)
        x, y = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        dist = (x - cx)**2 + (y - cy)**2
        # Fuzzy boundary mask
        mask = ((dist < 250) & (torch.rand(64, 64) > 0.25)).long()
        masks[i] = mask
        images[i, 0] += mask.float() * 1.2
        images[i, 1] += torch.sin(x.float()/5.0) * mask.float()
    return images.to(device), masks.to(device)


def run_segmentation_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Hard UNet Segmentation Benchmark on {device}...")

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    seeds = [42, 123]
    results = {}

    print("\n================ Hard Semantic Segmentation (Mean IoU ↑) ================")
    for act_type in activations:
        mious = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            model = UNet(n_channels=3, n_classes=2, act_type=act_type).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for step in range(300):
                xb, yb = generate_hard_batch(batch_size=32, device=device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_ious = []
            with torch.no_grad():
                for _ in range(20):
                    xb, yb = generate_hard_batch(batch_size=32, device=device)
                    logits = model(xb)
                    miou = compute_miou(logits, yb)
                    val_ious.append(miou)

            mean_val_miou = np.mean(val_ious)
            mious.append(mean_val_miou)
            print(f"[{act_type.upper():<14} | Seed {s}] Validation Mean IoU: {mean_val_miou:.4f}")

        results[act_type] = (np.mean(mious), np.std(mious))

    print("\n================ HARD SEGMENTATION SUMMARY ================")
    for act_type, (mean_miou, std_miou) in results.items():
        print(f"  {act_type.upper():<14}: Mean IoU = {mean_miou:.4f} ± {std_miou:.4f}")

if __name__ == '__main__':
    run_segmentation_benchmark()
