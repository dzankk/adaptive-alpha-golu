"""
Benchmark: Dense Object Detection (Mini-RetinaNet + FPN)
Benchmarking activation dynamics across multi-scale feature pyramids (FPN) 
and parallel multi-head architectures (classification vs. bounding box regression).
Calculates combined loss (Cross-Entropy + Smooth L1) across activation variants.
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
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))

class AdaptiveAlphaGoLU(nn.Module):
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))

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
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    elif act_type == 'swish_adaptive':
        return SwishAdaptive(init_beta=1.0)
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
# 2. Object Detection Architecture
# ==========================================
class FPN(nn.Module):
    def __init__(self, act_type='relu'):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), get_activation(act_type))
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), get_activation(act_type))
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
class SyntheticDetectionDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        img = torch.randn(3, 64, 64)
        # Target classes for 16x16 spatial map with 3 anchors per location
        cls_target = torch.randint(0, 5, (3, 16, 16), dtype=torch.long)
        box_target = torch.randn(12, 16, 16)
        return img, cls_target, box_target

def run_detection_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Detection Benchmark on {device}")

    loader = DataLoader(SyntheticDetectionDataset(), batch_size=8, shuffle=True)
    activations = ['relu', 'gelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        model = MiniRetinaNet(act_type=act_type).to(device)
        optimizer = get_optimizer(model)
        cls_loss_fn = nn.CrossEntropyLoss()
        box_loss_fn = nn.SmoothL1Loss()

        model.train()
        total_loss = 0.0
        for epoch in range(2):
            for img, cls_target, box_target in loader:
                img, cls_target, box_target = img.to(device), cls_target.to(device), box_target.to(device)
                optimizer.zero_grad()
                
                cls_logits, box_preds = model(img)
                
                # Reshape cls_logits from (B, 15, 16, 16) -> (B * 3, 5, 16, 16) to align with CrossEntropyLoss
                B, C_total, H, W = cls_logits.shape
                cls_logits_reshaped = cls_logits.view(B, 3, 5, H, W).view(-1, 5, H, W)
                cls_target_reshaped = cls_target.view(-1, H, W)
                
                loss_cls = cls_loss_fn(cls_logits_reshaped, cls_target_reshaped)
                loss_box = box_loss_fn(box_preds, box_target)
                
                loss = loss_cls + loss_box
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(loader) * 2)
        print(f"Activation: {act_type.ljust(15)} | Validation Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    run_benchmark()
