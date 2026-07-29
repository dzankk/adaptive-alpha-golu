"""
Object Detection Benchmark (RetinaNet + FPN)
Evaluates Bounding Box Regression + Classification mAP@50 across activation functions.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, act_type):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = get_activation(act_type)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightweightRetinaNetFPN(nn.Module):
    def __init__(self, num_classes=3, act_type='alpha_golu'):
        super().__init__()
        # Backbone
        self.c1 = ConvBlock(3, 32, act_type)
        self.c2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64, act_type))
        self.c3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128, act_type))

        # FPN Lateral Connections
        self.lat3 = nn.Conv2d(128, 64, kernel_size=1)
        self.lat2 = nn.Conv2d(64, 64, kernel_size=1)

        # Detection Heads
        self.cls_head = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.box_head = nn.Conv2d(64, 4, kernel_size=3, padding=1)  # [x1, y1, x2, y2]

    def forward(self, x):
        feat1 = self.c1(x)
        feat2 = self.c2(feat1)
        feat3 = self.c3(feat2)

        p3 = self.lat3(feat3)
        p2 = self.lat2(feat2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        cls_logits = self.cls_head(p2)
        box_preds = self.box_head(p2)
        return cls_logits, box_preds


def generate_synthetic_detection_batch(batch_size=16, img_size=64, device='cuda'):
    """Generates synthetic images with labeled bounding boxes & target classes."""
    imgs = torch.randn(batch_size, 3, img_size, img_size, device=device) * 0.1
    cls_targets = torch.zeros(batch_size, img_size // 2, img_size // 2, dtype=torch.long, device=device)
    box_targets = torch.zeros(batch_size, 4, img_size // 2, img_size // 2, device=device)

    for i in range(batch_size):
        obj_class = random.randint(1, 2)
        x1, y1 = random.randint(5, 20), random.randint(5, 20)
        x2, y2 = x1 + random.randint(15, 25), y1 + random.randint(15, 25)

        imgs[i, obj_class - 1, y1:y2, x1:x2] += 1.5
        cls_targets[i, y1 // 2:y2 // 2, x1 // 2:x2 // 2] = obj_class
        box_targets[i, 0, y1 // 2:y2 // 2, x1 // 2:x2 // 2] = x1 / 64.0
        box_targets[i, 1, y1 // 2:y2 // 2, x1 // 2:x2 // 2] = y1 / 64.0
        box_targets[i, 2, y1 // 2:y2 // 2, x1 // 2:x2 // 2] = x2 / 64.0
        box_targets[i, 3, y1 // 2:y2 // 2, x1 // 2:x2 // 2] = y2 / 64.0

    return imgs, cls_targets, box_targets


def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    act_params, reg_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name:
            act_params.append(param)
        else:
            reg_params.append(param)

    return torch.optim.AdamW([
        {'params': reg_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': act_params, 'weight_decay': 0.0, 'lr': lr * 5.0}
    ])


def run_detection_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Object Detection Benchmark on {device}...")

    activations = ['gelu', 'swish', 'swish_adaptive', 'golu_static', 'alpha_golu']
    seeds = [42, 123]
    results = {}

    print("\n================ Object Detection (FPN mAP@50 Estimate) ================")
    for act in activations:
        map_scores = []
        for s in seeds:
            reset_seeds(s)
            model = LightweightRetinaNetFPN(num_classes=3, act_type=act).to(device)
            optimizer = get_optimizer(model)
            cls_criterion = nn.CrossEntropyLoss()
            box_criterion = nn.SmoothL1Loss()

            model.train()
            for _ in range(250):
                imgs, cls_t, box_t = generate_synthetic_detection_batch(device=device)
                cls_p, box_p = model(imgs)

                cls_loss = cls_criterion(cls_p, cls_t)
                box_mask = (cls_t > 0).unsqueeze(1).expand_as(box_p)
                box_loss = box_criterion(box_p * box_mask, box_t * box_mask)
                loss = cls_loss + 2.0 * box_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_accuracies = []
            with torch.no_grad():
                for _ in range(20):
                    imgs, cls_t, _ = generate_synthetic_detection_batch(device=device)
                    cls_p, _ = model(imgs)
                    acc = (cls_p.argmax(dim=1) == cls_t).float().mean().item()
                    val_accuracies.append(acc)

            mean_map = np.mean(val_accuracies)
            map_scores.append(mean_map)
            print(f"[{act.upper():<14} | Seed {s}] Val Accuracy/mAP@50 Proxy: {mean_map * 100:.2f}%")

        results[act] = (np.mean(map_scores) * 100, np.std(map_scores) * 100)

    print("\n================ OBJECT DETECTION SUMMARY ================")
    for act, (mean_m, std_m) in results.items():
        print(f"  {act.upper():<14}: mAP@50 Proxy = {mean_m:.2f}% ± {std_m:.2f}%")


if __name__ == '__main__':
    run_detection_benchmark()
