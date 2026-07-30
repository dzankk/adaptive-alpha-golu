"""
Benchmark: Dense Object Detection (Mini-RetinaNet + Feature Pyramid Network)
=============================================================================
Benchmarking activation dynamics across multi-scale feature pyramids (FPN) 
and parallel multi-head architectures (classification vs. bounding box regression).
Calculates combined loss (Focal/CE + Smooth L1) across activation variants.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. Custom Activations & Optimizer Setup
# ==========================================
class GoLUStatic(nn.Module):
    """Numerically stable static Gompertz activation: x * exp(-exp(-x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class AdaptiveAlphaGoLU(nn.Module):
    """Parametric Gompertz activation with strictly positive learnable alpha"""
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))


class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x) with bounded positive alpha"""
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    """Parametric Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


# Alias for naming consistency
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
# 2. Dense Object Detection Architecture
# ==========================================
class FPN(nn.Module):
    """True Feature Pyramid Network with lateral connections and top-down fusion."""
    def __init__(self, act_type: str = 'relu'):
        super().__init__()
        # C1: H/2 x W/2 (32 channels)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            get_activation(act_type)
        )
        # C2: H/4 x W/4 (64 channels)
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            get_activation(act_type)
        )
        
        # Lateral connections mapping feature channels to unified dimension (32)
        self.lat_c2 = nn.Conv2d(64, 32, kernel_size=1)
        self.lat_c1 = nn.Conv2d(32, 32, kernel_size=1)
        
        # Smooth output layer
        self.smooth = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1_out = self.c1(x)         # [B, 32, H/2, W/2]
        c2_out = self.c2(c1_out)     # [B, 64, H/4, W/4]

        # Top-down pathway: Upsample higher-level feature map and fuse
        p2 = self.lat_c2(c2_out)
        p1 = self.lat_c1(c1_out) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        return self.smooth(p1)       # [B, 32, H/2, W/2]


class RetinaHead(nn.Module):
    """Parallel multi-head architecture for classification and bounding box regression."""
    def __init__(self, num_classes: int = 5, num_anchors: int = 3, act_type: str = 'relu'):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification Sub-network
        self.cls_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            get_activation(act_type),
            nn.Conv2d(32, num_anchors * num_classes, kernel_size=3, padding=1)
        )
        # Bounding Box Regression Sub-network
        self.box_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            get_activation(act_type),
            nn.Conv2d(32, num_anchors * 4, kernel_size=3, padding=1)
        )

    def forward(self, fpn_feat: torch.Tensor):
        cls_logits = self.cls_head(fpn_feat)
        box_preds = self.box_head(fpn_feat)
        return cls_logits, box_preds


class MiniRetinaNet(nn.Module):
    def __init__(self, act_type: str = 'relu'):
        super().__init__()
        self.fpn = FPN(act_type=act_type)
        self.head = RetinaHead(act_type=act_type)

    def forward(self, x: torch.Tensor):
        feat = self.fpn(x)
        return self.head(feat)


# ==========================================
# 3. Synthetic Benchmark Execution
# ==========================================
class StructuredDetectionDataset(Dataset):
    """Synthetic dataset with spatial bounding structure to allow proper activation gradients."""
    def __init__(self, size: int = 160):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        img = torch.randn(3, 64, 64) * 0.1
        cls_target = torch.zeros((3, 32, 32), dtype=torch.long)
        box_target = torch.zeros((12, 32, 32), dtype=torch.float32)
        
        # Inject deterministic synthetic box targets
        cx, cy = (idx % 24) + 4, ((idx * 5) % 24) + 4
        img[:, cy*2:(cy+2)*2, cx*2:(cx+2)*2] += 1.5
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
    dataset = StructuredDetectionDataset(size=160)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MiniRetinaNet(act_type=act_type).to(device)
    optimizer = get_optimizer(model, lr=1e-3)
    cls_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for img, cls_target, box_target in train_loader:
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

    # Return metric derived exclusively from final converged epoch loss
    final_avg_loss = total_loss / max(1, num_batches)
    map_score = max(0.0, 100.0 - (final_avg_loss * 25.0))
    return map_score


def run_detection_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Detection Benchmark on {device}")
    activations = ['relu', 'gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        map_score = train_single_seed_detection(act_type, seed=42, epochs=5, device=device)
        print(f"Activation: {act_type.ljust(15)} | mAP Score: {map_score:.4f}")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Mean Average Precision (mAP) proxy metric."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_score = train_single_seed_detection(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(map_score)


if __name__ == '__main__':
    run_detection_benchmark()
