"""
Alpha-Gompertz Linear Unit (Alpha-GoLU) Module
==============================================
This module implements standard (Static) GoLU and the parameterized 
Adaptive Gompertz Linear Unit (Alpha-GoLU). Alpha-GoLU parameterizes the 
double-exponential Gumbel CDF gate, allowing backpropagation to autonomously 
optimize layer-wise asymmetry and latent variance squeezing boundaries.

Author: Džana Kopić
Paper Reference: Gompertz Linear Units (Das et al., 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentVarianceTracker(nn.Module):
    """
    Wrapper module to record post-activation latent variances 
    for empirical logging and variance analysis.
    """
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation
        self.register_buffer("last_output_var", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        if self.training:
            with torch.no_grad():
                self.last_output_var = out.detach().var().cpu()
        return out


class StaticGoLU(nn.Module):
    """
    Standard unparameterized Gompertz Linear Unit (Das et al., 2025).
    
    Mathematical Formulation:
        GoLU(x) = x * exp(-exp(-x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = torch.clamp(-x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(scaled_x))
        return x * gate


class AlphaGoLU(nn.Module):
    """
    Parameterized Adaptive Gompertz Linear Unit (Alpha-GoLU).
    
    Mathematical Formulation:
        AlphaGoLU(x) = x * exp(-exp(-alpha * x))
        where alpha = softplus(raw_alpha) > 0
    """
    def __init__(self, num_parameters: int = 1, init_alpha: float = 1.0):
        super().__init__()
        self.num_parameters = num_parameters
        
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        
        self.raw_alpha = nn.Parameter(
            torch.full((num_parameters,), init_raw, dtype=torch.float32)
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Guarantees alpha is strictly positive (alpha > 0)."""
        return F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_alpha = self.alpha

        if self.num_parameters > 1:
            if x.dim() == 4:     # CNN Feature Map (B, C, H, W)
                shape = (1, -1, 1, 1)
            elif x.dim() == 3:   # Transformer Sequence (B, L, C)
                shape = (1, 1, -1)
            else:                # Standard Linear Output (B, C)
                shape = (1, -1)
            alpha_param = current_alpha.view(*shape)
        else:
            alpha_param = current_alpha

        scaled_x = torch.clamp(-alpha_param * x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(scaled_x))
        return x * gate

    def get_alpha_val(self) -> torch.Tensor:
        """Returns the current alpha parameter tensor for tracking/logging."""
        return self.alpha.detach()

    def extra_repr(self) -> str:
        current_a = self.alpha.detach()
        if self.num_parameters == 1:
            return f"num_parameters=1, current_alpha={current_a.item():.4f}"
        return f"num_parameters={self.num_parameters}, mean_alpha={current_a.mean().item():.4f}"


AdaptiveAlphaGoLU = AlphaGoLU
