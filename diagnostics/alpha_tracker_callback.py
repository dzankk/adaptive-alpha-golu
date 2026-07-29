"""
Alpha Parameter Trajectory Logger
=================================
Utility module to extract, record, and visualize learned alpha values across 
network depth and training epochs.

Author: Your Research Team
"""

import torch
import torch.nn as nn
from typing import Dict, List
import matplotlib.pyplot as plt
from alpha_golu import AlphaGoLU


class AlphaTrajectoryLogger:
    """
    Hooks into PyTorch models to track the evolution of alpha across layers.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.alpha_history: Dict[str, List[float]] = {}
        self._register_alpha_layers()

    def _register_alpha_layers(self):
        """Finds all AlphaGoLU layers in the model and registers their names."""
        for name, module in self.model.named_modules():
            if isinstance(module, AlphaGoLU):
                self.alpha_history[name] = []

    def step(self):
        """Appends current alpha values to history (call at end of epoch/step)."""
        for name, module in self.model.named_modules():
            if isinstance(module, AlphaGoLU):
                mean_alpha = module.get_alpha_val().mean().item()
                self.alpha_history[name].append(mean_alpha)

    def plot_trajectories(self, save_path: str = "alpha_trajectories.png"):
        """Plots learned alpha values over time across network depth."""
        plt.figure(figsize=(10, 6))
        for layer_name, history in self.alpha_history.items():
            plt.plot(history, label=layer_name)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Static GoLU baseline (alpha=1.0)')
        plt.xlabel("Training Step / Epoch")
        plt.ylabel(r"Learned Parameter $\alpha$")
        plt.title(r"Evolution of Alpha ($\alpha$) Across Network Depth")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
