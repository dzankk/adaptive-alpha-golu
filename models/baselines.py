"""
Standardized Benchmark Networks
================================
Provides configurable deep architectures (CNN and MLP) for multi-dataset 
comparative evaluations under fixed parameter initialization seeds.
"""

import math
import torch
import torch.nn as nn
from models.alpha_golu import StaticGoLU, AlphaGoLU, LatentVarianceTracker


def get_activation_layer(act_name: str, init_alpha: float = 1.0) -> nn.Module:
    """Factory function for instantiating activation functions across all experiment runners."""
    act_name = act_name.lower()
    if act_name in ['gelu', 'gelu_static']:
        return nn.GELU()
    elif act_name in ['swish', 'silu']:
        return nn.SiLU()
    elif act_name == 'prelu':
        return nn.PReLU(init=init_alpha)
    elif act_name == 'pgelu':
        return ParametricGELU(init_alpha=init_alpha)
    elif act_name == 'golu_static':
        return StaticGoLU()
    elif act_name in ['alpha_golu', 'adaptive_alpha_golu']:
        return AlphaGoLU(init_alpha=init_alpha)  # or AdaptiveAlphaGoLU(init_alpha=init_alpha)
    elif act_name in ['adaptive_swish', 'swish_adaptive']:
        return AdaptiveSwish(init_beta=init_alpha)
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")


class ParametricGELU(nn.Module):
    """
    Parametric GELU (P-GELU): x * Phi(alpha * x)
    Uses a learnable alpha scalar per layer to scale Gaussian CDF.
    """
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha)

    def get_alpha_val(self) -> torch.Tensor:
        return self.alpha.detach()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356))


class AdaptiveSwish(nn.Module):
    """Adaptive Swish (SiLU): x * sigmoid(beta * x)."""

    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def get_alpha_val(self) -> torch.Tensor:
        return self.beta.detach()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class DeepConvNet(nn.Module):
    """
    Deep Convolutional Network featuring 4 sequential feature processing blocks.
    Tracks layer-wise alpha parameters and latent variance across depth.
    """
    def __init__(self, act_type: str = 'alpha_golu', num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.act_type = act_type.lower()
        
        self.act1 = get_activation_layer(act_type)
        self.act2 = get_activation_layer(act_type)
        self.act3 = get_activation_layer(act_type)
        self.act4 = get_activation_layer(act_type)
        
        self.tr1 = LatentVarianceTracker(self.act1)
        self.tr2 = LatentVarianceTracker(self.act2)
        self.tr3 = LatentVarianceTracker(self.act3)
        self.tr4 = LatentVarianceTracker(self.act4)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.tr1,
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.tr2,
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.tr3,
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.tr4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def extract_alphas(self) -> list:
        """Returns the learned activation parameters present in the network."""
        extracted = []
        for module in [self.act1, self.act2, self.act3, self.act4]:
            if hasattr(module, "get_alpha_val"):
                extracted.append(module.get_alpha_val())
            elif hasattr(module, "alpha"):
                alpha_value = module.alpha
                if isinstance(alpha_value, torch.Tensor):
                    extracted.append(alpha_value.detach().cpu())
                else:
                    extracted.append(torch.tensor(float(alpha_value)))
            elif hasattr(module, "beta"):
                extracted.append(module.beta.detach().cpu())
            elif isinstance(module, nn.PReLU):
                extracted.append(module.weight.detach().cpu())
        return extracted

    def extract_latent_variances(self) -> list:
        """Returns the latest post-activation variances across layers."""
        return [
            self.tr1.last_output_var,
            self.tr2.last_output_var,
            self.tr3.last_output_var,
            self.tr4.last_output_var
        ]
