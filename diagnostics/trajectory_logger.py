"""
Alpha Parameter Trajectory Logger
=================================
Utility module to extract, record, and plot learned alpha values across 
network depth over training epochs.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List
from models.alpha_golu import AlphaGoLU


class AlphaTrajectoryLogger:
    """Hooks into PyTorch models to track layer-wise alpha parameter evolution."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.alpha_history: Dict[str, List[float]] = {}
        self._register_alpha_layers()

    def _register_alpha_layers(self):
        """Finds all AlphaGoLU layers in the network."""
        for name, module in self.model.named_modules():
            if isinstance(module, AlphaGoLU):
                self.alpha_history[name] = []

    def step(self):
        """Records current alpha values at the end of an epoch or step."""
        for name, module in self.model.named_modules():
            if isinstance(module, AlphaGoLU):
                mean_alpha = module.get_alpha_val().mean().item()
                self.alpha_history[name].append(mean_alpha)

    def plot_trajectories(self, save_path: str = "alpha_trajectories.png"):
        """Plots learned alpha values across time and layer depth."""
        plt.figure(figsize=(10, 6))
        for layer_name, history in self.alpha_history.items():
            plt.plot(history, label=layer_name, marker='o', markersize=3)
            
        plt.axhline(y=1.0, color='r', linestyle='--', label='Static GoLU Baseline (1.0)')
        plt.xlabel("Training Step / Epoch")
        plt.ylabel(r"Learned Parameter $\alpha$")
        plt.title(r"Evolution of Alpha ($\alpha$) Across Network Depth")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Diagnostics] Alpha trajectory plot saved to: {save_path}")
