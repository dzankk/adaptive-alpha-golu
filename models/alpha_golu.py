"""
Alpha-Gompertz Linear Unit (Alpha-GoLU) Module
==============================================
This module implements both standard (Static) GoLU and the parameterized 
Adaptive Gompertz Linear Unit (Alpha-GoLU). Alpha-GoLU parameterizes the 
double-exponential Gumbel CDF gate, allowing backpropagation to autonomously 
optimize layer-wise asymmetry and latent variance squeezing boundaries.

Author: Your Research Team
Paper Reference: Gompertz Linear Units (Das et al., 2025)
"""

import torch
import torch.nn as nn


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
        # Clamp exponent to prevent numerical float32 overflow/underflow
        scaled_x = torch.clamp(-x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(scaled_x))
        return x * gate


class AlphaGoLU(nn.Module):
    """
    Parameterized Adaptive Gompertz Linear Unit (Alpha-GoLU).
    
    Mathematical Formulation:
        AlphaGoLU(x) = x * exp(-exp(-alpha * x))
        
    Attributes:
        num_parameters (int): Number of learnable alpha parameters. 
            1 for layer-wide scalar, or C for channel-wise vectors.
        init_alpha (float): Initial value for parameter alpha (Default: 1.0).
        alpha (nn.Parameter): Learnable scaling factor governing gate asymmetry.
    """
    def __init__(self, num_parameters: int = 1, init_alpha: float = 1.0):
        super().__init__()
        self.num_parameters = num_parameters
        self.alpha = nn.Parameter(
            torch.full((num_parameters,), float(init_alpha), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast channel-wise vs. scalar parameters
        if self.num_parameters > 1:
            if x.dim() == 4:     # CNN Feature Map (B, C, H, W)
                shape = (1, -1, 1, 1)
            elif x.dim() == 3:   # Transformer Sequence (B, L, C)
                shape = (1, 1, -1)
            else:               # Standard Linear Output (B, C)
                shape = (1, -1)
            alpha_param = self.alpha.view(*shape)
        else:
            alpha_param = self.alpha

        # Compute double-exponential Gompertz gate with safe clamping
        scaled_x = torch.clamp(-alpha_param * x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(scaled_x))
        return x * gate

    def get_alpha_val(self) -> torch.Tensor:
        """Returns the current alpha parameter tensor for logging."""
        return self.alpha.detach()

    def extra_repr(self) -> str:
        if self.num_parameters == 1:
            return f"num_parameters=1, current_alpha={self.alpha.item():.4f}"
        return f"num_parameters={self.num_parameters}, mean_alpha={self.alpha.mean().item():.4f}""""
Alpha-Gompertz Linear Unit (Alpha-GoLU) Module
==============================================
This module implements both standard (Static) GoLU and the parameterized 
Adaptive Gompertz Linear Unit (Alpha-GoLU). Alpha-GoLU parameterizes the 
double-exponential Gumbel CDF gate, allowing backpropagation to autonomously 
optimize layer-wise asymmetry and latent variance squeezing boundaries.

Author: Your Research Team
Paper Reference: Gompertz Linear Units (Das et al., 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGoLU(nn.Module):
    """
    Standard unparameterized Gompertz Linear Unit (Das et al., 2025).
    
    Mathematical Formulation:
        GoLU(x) = x * exp(-exp(-x))
    """
    def __init__(self):
        super(StaticGoLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamped inside exp to prevent numerical overflow in double-exp
        scaled_x = torch.clamp(x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(-scaled_x))
        return x * gate


class AlphaGoLU(nn.Module):
    """
    Parameterized Adaptive Gompertz Linear Unit (Alpha-GoLU).
    
    Mathematical Formulation:
        AlphaGoLU(x) = x * exp(-exp(-alpha * x))
        
    Attributes:
        num_parameters (int): Number of learnable alpha parameters. 
            1 for layer-wide scalar, or C for channel-wise vectors.
        init_alpha (float): Initial value for the parameter alpha (Default: 1.0).
        alpha (nn.Parameter): Learnable scaling factor governing gate asymmetry.
    """
    def __init__(self, num_parameters: int = 1, init_alpha: float = 1.0):
        super(AlphaGoLU, self).__init__()
        self.num_parameters = num_parameters
        # Initialize alpha as a learnable parameter tensor
        self.alpha = nn.Parameter(
            torch.full((num_parameters,), float(init_alpha), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle dimensional broadcasting for channel-wise vs layer-wide parameters
        if self.num_parameters > 1:
            if x.dim() == 4:     # CNN Feature Map (B, C, H, W)
                shape = (1, -1, 1, 1)
            elif x.dim() == 3:   # Transformer Sequence (B, L, C)
                shape = (1, 1, -1)
            else:               # Standard Linear Output (B, C)
                shape = (1, -1)
            alpha_param = self.alpha.view(*shape)
        else:
            alpha_param = self.alpha

        # Scaled input with numerical safety bounds (fp32 double-exp limit is ~88.0)
        scaled_x = torch.clamp(-alpha_param * x, min=-88.0, max=88.0)
        gate = torch.exp(-torch.exp(scaled_x))
        return x * gate

    def get_alpha_val(self) -> torch.Tensor:
        """
        Returns the current alpha parameter tensor for tracking/logging.
        """
        return self.alpha.detach()

    def extra_repr(self) -> str:
        if self.num_parameters == 1:
            return f"num_parameters=1, current_alpha={self.alpha.item():.4f}"
        return f"num_parameters={self.num_parameters}, mean_alpha={self.alpha.mean().item():.4f}"
