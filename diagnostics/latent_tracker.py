"""
Latent Variance Diagnostic Tracker
==================================
Wrapper module designed to log activation statistics (variance and mean)
before and after passing through an activation layer to empirical test the 
Variance Reduction Hypothesis.
"""

import torch
import torch.nn as nn


class LatentVarianceTracker(nn.Module):
    """
    Diagnostic wrapper logging activation variance (sigma^2) and mean
    before and after passing through an activation module.
    """
    def __init__(self, activation_module: nn.Module):
        super().__init__()
        self.activation = activation_module
        self.last_input_var = 0.0
        self.last_output_var = 0.0
        self.last_input_mean = 0.0
        self.last_output_mean = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.last_input_var = float(torch.var(x).item())
                self.last_input_mean = float(torch.mean(x).item())
        
        out = self.activation(x)
        
        if self.training:
            with torch.no_grad():
                self.last_output_var = float(torch.var(out).item())
                self.last_output_mean = float(torch.mean(out).item())
            
        return out

    def get_variance_reduction_ratio(self) -> float:
        """Calculates Var(out) / Var(in). Values < 1.0 indicate compression."""
        if self.last_input_var == 0.0:
            return 1.0
        return self.last_output_var / self.last_input_var
