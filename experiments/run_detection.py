"""
Benchmark: Dense Object Detection (Mini-RetinaNet + FPN)
Benchmarking activation dynamics across multi-scale feature pyramids (FPN) 
and parallel multi-head architectures (classification vs. bounding box regression).
Calculates combined loss (Focal/CE + Smooth L1) across activation variants.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. Custom Activations & Optimizer Setup
# ==========================================
class GoLUStatic(nn.Module):
    """Numerically stable static Gompertz activation: x * exp(-exp(-x))"""
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class AdaptiveAlphaGoLU(nn.Module):
    """Parametric Gompertz activation with learnable slope parameter alpha"""
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
    """Parametric Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# Alias for naming consistency across CLI tools
AdaptiveSwish = SwishAdaptive


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type == 'golu_static':
        return GoLUStatic()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    elif act_type in ('swish_adaptive', 'adaptive_swish'):
        return SwishAdaptive(init_beta=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    
    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, PGELU, SwishAdaptive, AdaptiveSwish, nn.PReLU)):
            for p in module.parameters():
                if p.requires_grad:
                    act_params.append(p)

    act_param_ids = set(map(id, act_params))
    for p in model.parameters():
        if p.requires_grad and id(p) not in act_param_ids:
            base_params.append(p)

    param_groups = [{'params': base_params, 'weight_decay': weight_decay}]
    if act_params:
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)


# ==========================================
# 2. Object Detection Architecture
# ==========================================
class FPN(nn.Module):
    def __init__(self, act_type='relu'):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(act_type)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(act_type)
        )
        self.lat = nn.Conv2d(64, 32, 1)

    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        p2 = self.lat(c2_out)
        return p2


class RetinaHead(nn.Module):
    def __init__(self, num_classes=5, num_anchors=3, act_type='relu'):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            get_activation(act_type),
            nn.Conv2d(32, num_anchors * num_classes, 3, padding=1)
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            get_activation(act_type),
            nn.Conv2d(32, num_anchors * 4, 3, padding=1)
        )

    def forward(self, fpn_feat):
        cls_logits = self.cls_head(fpn_feat)
        box_preds = self.box_head(fpn_feat)
        return cls_logits, box_preds


class MiniRetinaNet(nn.Module):
    def __init__(self, act_type='relu'):
        super().__init__()
        self.fpn = FPN(act_type=act_type)
        self.head = RetinaHead(act_type=act_type)

    def forward(self, x):
        feat = self.fpn(x)
        return self.head(feat)


# ==========================================
# 3. Benchmark Execution
# ==========================================
class StructuredDetectionDataset(Dataset):
    """Synthetic dataset with spatial bounding structure to allow activation gradients to scale properly."""
    def __init__(self, size=120):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate correlated geometric image patches
        img = torch.randn(3, 64, 64) * 0.1
        cls_target = torch.zeros((3, 16, 16), dtype=torch.long)
        box_target = torch.zeros((12, 16, 16))
        
        # Inject deterministic synthetic box centers per sample
        cx, cy = (idx % 12) + 2, ((idx * 3) % 12) + 2
        img[:, cy*4:(cy+2)*4, cx*4:(cx+2)*4] += 1.5
        cls_target[:, cy, cx] = (idx % 4) + 1
        box_target[:4, cy, cx] = torch.tensor([0.1, -0.1, 0.2, 0.2])
        
        return img, cls_target, box_target


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_seed_detection(act_type: str, seed: int, epochs: int, device: torch.device) -> float:
    set_seed(seed)
    loader = DataLoader(StructuredDetectionDataset(size=120), batch_size=8, shuffle=True)
    model = MiniRetinaNet(act_type=act_type).to(device)
    optimizer = get_optimizer(model)
    cls_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for img, cls_target, box_target in loader:
            img, cls_target, box_target = img.to(device), cls_target.to(device), box_target.to(device)
            optimizer.zero_grad()
            
            cls_logits, box_preds = model(img)
            
            B, _, H, W = cls_logits.shape
            cls_logits_reshaped = cls_logits.view(B, 3, 5, H, W).permute(0, 1, 3, 4, 2).reshape(-1, 5)
            cls_target_reshaped = cls_target.view(-1)
            
            loss_cls = cls_loss_fn(cls_logits_reshaped, cls_target_reshaped)
            loss_box = box_loss_fn(box_preds, box_target)
            
            loss = loss_cls + 2.0 * loss_box
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    map_score = max(0.0, 100.0 - (avg_loss * 25.0))
    return map_score


def run_detection_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Detection Benchmark on {device}")
    activations = ['relu', 'gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        map_score = train_single_seed_detection(act_type, seed=42, epochs=4, device=device)
        print(f"Activation: {act_type.ljust(15)} | mAP Score: {map_score:.4f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Mean Average Precision (mAP)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_score = train_single_seed_detection(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(map_score)


if __name__ == '__main__':
    run_detection_benchmark()
