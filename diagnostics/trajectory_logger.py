"""
Parametric Activation Trajectory Logger
=======================================
Utility module to extract, record, and plot learned parameter values across 
network depth over training epochs.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List
from models.alpha_golu import AlphaGoLU


_TRACKABLE_NAME_HINTS = (
    "alphagolu",
    "golu",
    "pgelu",
    "parametricgelu",
    "swish",
    "swishadaptive",
    "adaptiveswish",
    "prelu",
)


class AlphaTrajectoryLogger:
    """Hooks into PyTorch models to track layer-wise parametric activation evolution."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.alpha_history: Dict[str, List[float]] = {}
        self._register_alpha_layers()

    def _is_trackable_module(self, module: nn.Module) -> bool:
        if isinstance(module, nn.PReLU):
            return True

        module_name = module.__class__.__name__.lower()
        return any(hint in module_name for hint in _TRACKABLE_NAME_HINTS)

    def _extract_module_value(self, module: nn.Module) -> torch.Tensor | None:
        if hasattr(module, "get_alpha_val"):
            value = module.get_alpha_val()
            return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(float(value))

        if hasattr(module, "alpha"):
            value = module.alpha
            return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(float(value))

        if hasattr(module, "beta"):
            value = module.beta
            return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(float(value))

        if isinstance(module, nn.PReLU):
            return module.weight.detach().cpu()

        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if name in {"raw_alpha", "beta", "weight"}:
                return parameter.detach().cpu()

        return None

    def _register_alpha_layers(self):
        """Finds all parametric activation layers in the network."""
        for name, module in self.model.named_modules():
            if isinstance(module, AlphaGoLU) or self._is_trackable_module(module):
                self.alpha_history[name] = []

    def step(self):
        """Records current parameter values at the end of an epoch or step."""
        for name, module in self.model.named_modules():
            if name not in self.alpha_history:
                continue
            value = self._extract_module_value(module)
            if value is None:
                continue
            self.alpha_history[name].append(float(value.mean().item()))

    def plot_trajectories(self, save_path: str = "alpha_trajectories.png"):
        """Plots learned parameter values across time and layer depth."""
        plt.figure(figsize=(10, 6))
        for layer_name, history in self.alpha_history.items():
            plt.plot(history, label=layer_name, marker='o', markersize=3)
            
        plt.axhline(y=1.0, color='r', linestyle='--', label='Reference Value (1.0)')
        plt.xlabel("Training Step / Epoch")
        plt.ylabel(r"Learned Parameter")
        plt.title(r"Evolution of Parametric Activation Values Across Network Depth")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Diagnostics] Alpha trajectory plot saved to: {save_path}")
