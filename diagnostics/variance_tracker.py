"""
Latent Variance Diagnostic Tracker
==================================
Wrapper module designed to track activation statistics (variance and mean)
before and after passing through an activation layer. Used to verify the 
Variance Reduction Hypothesis during training runs.

Author: Your Research Team
"""

import torch
import torch.nn as nn


class LatentVarianceTracker(nn.Module):
    """
    Diagnostic wrapper module that logs activation variance (sigma^2)
    and mean before/after passing through an activation function.
    
    Attributes:
        activation (nn.Module): The underlying activation module (e.g., AlphaGoLU).
        last_input_var (float): Last recorded input activation variance.
        last_output_var (float): Last recorded output activation variance.
        last_input_mean (float): Last recorded input activation mean.
        last_output_mean (float): Last recorded output activation mean.
    """
    def __init__(self, activation_module: nn.Module):
        super(LatentVarianceTracker, self).__init__()
        self.activation = activation_module
        self.last_input_var = 0.0
        self.last_output_var = 0.0
        self.last_input_mean = 0.0
        self.last_output_mean = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Record input statistics without altering autograd graph
        with torch.no_grad():
            self.last_input_var = torch.var(x).item()
            self.last_input_mean = torch.mean(x).item()
        
        # Pass through actual activation layer
        out = self.activation(x)
        
        # Record output statistics
        with torch.no_grad():
            self.last_output_var = torch.var(out).item()
            self.last_output_mean = torch.mean(out).item()
            
        return out

    def get_variance_reduction_ratio(self) -> float:
        """
        Calculates the ratio Var(out) / Var(in). 
        Ratios < 1.0 indicate variance compression.
        """
        if self.last_input_var == 0.0:
            return 1.0
        return self.last_output_var / self.last_input_var
