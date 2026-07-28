"""
Adaptive Asymmetry Gompertz Linear Unit (Alpha-GoLU)
===================================================
This module defines the architectural advancement extending the Gompertz Linear
Unit (GoLU) proposed by Das et al. (2025). It parameterizes the double-exponential 
Gumbel CDF gate, allowing backpropagation to autonomously optimize layer-wise 
asymmetry and latent variance squeezing boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGoLU(nn.Module):
    """
    Standard unparameterized Gompertz Linear Unit (Das et al., 2025).
    Formula: GoLU(x) = x * exp(-exp(-x))
    """
    def __init__(self):
        super(StaticGoLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamped inside exp to prevent numerical overflow in double-exp
        gate = torch.exp(-torch.exp(-torch.clamp(x, min=-10.0, max=10.0)))
        return x * gate


class AlphaGoLU(nn.Module):
    """
    Parameterized Adaptive Gompertz Linear Unit.
    Formula: GoLU_alpha(x) = x * exp(-exp(-alpha * x))
    
    Attributes:
        alpha (nn.Parameter): Learnable scaling factor governing gate asymmetry.
    """
    def __init__(self, init_alpha: float = 1.0):
        super(AlphaGoLU, self).__init__()
        # Alpha initialized as a single learnable scalar parameter
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaled input with numerical safety bounds
        scaled_x = torch.clamp(self.alpha * x, min=-10.0, max=10.0)
        gate = torch.exp(-torch.exp(-scaled_x))
        return x * gate

    def get_alpha_val(self) -> float:
        """Returns the scalar float value of alpha for tracking."""
        return self.alpha.item()

    def extra_repr(self) -> str:
        return f"init_alpha=1.0, current_alpha={self.alpha.item():.4f}"


class LatentVarianceTracker(nn.Module):
    """
    Diagnostic wrapper module that logs activation variance (sigma^2)
    and mean before/after passing through an activation function.
    """
    def __init__(self, activation_module: nn.Module):
        super(LatentVarianceTracker, self).__init__()
        self.activation = activation_module
        self.last_input_var = 0.0
        self.last_output_var = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.last_input_var = torch.var(x).item()
        
        out = self.activation(x)
        
        with torch.no_grad():
            self.last_output_var = torch.var(out).item()
            
        return out
