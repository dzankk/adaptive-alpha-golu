"""
Publication-Quality Visualization Suite
========================================
Generates vector-grade figures depicting trajectory dynamics, variance reduction,
and accuracy comparisons for inclusion in presentations and final reports.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_dashboard(results_dict: dict, save_dir: str = "outputs"):
    """
    Generates a 4-panel analysis dashboard.
    
    Panels:
    1. Validation Accuracy Convergence across Baselines
    2. Training Loss Curves
    3. Alpha Parameter Trajectories across Network Depth (Alpha-GoLU)
    4. Empirical Latent Space Variance (Sigma^2) Comparison
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Alpha-GoLU: Empirical Evaluation Dashboard", fontsize=16, fontweight='bold')

    # Panel 1: Validation Accuracy
    ax1 = axs[0, 0]
    for act_name, metrics in results_dict.items():
        ax1.plot(metrics['val_acc'], label=f"{act_name.upper()}", linewidth=2)
    ax1.set_title("Validation Accuracy Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Training Loss
    ax2 = axs[0, 1]
    for act_name, metrics in results_dict.items():
        ax2.plot(metrics['train_loss'], label=f"{act_name.upper()}", linewidth=2)
    ax2.set_title("Cross-Entropy Loss Trajectory")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Alpha Trajectories Across Layers (Alpha-GoLU only)
    ax3 = axs[1, 0]
    if 'alpha_golu' in results_dict and 'alpha_history' in results_dict['alpha_golu']:
        alpha_hist = np.array(results_dict['alpha_golu']['alpha_history']) # Shape: (epochs, num_layers)
        num_layers = alpha_hist.shape[1]
        for layer_idx in range(num_layers):
            ax3.plot(alpha_hist[:, layer_idx], marker='o', label=f"Layer {layer_idx+1} Alpha")
        ax3.axhline(1.0, color='red', linestyle='--', label='Static Baseline (1.0)')
        ax3.set_title("Alpha Parameter Evolution across Network Depth")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learned Alpha Value")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Alpha Tracking Inactive", ha='center', va='center')

    # Panel 4: Latent Space Variance Comparison
    ax4 = axs[1, 1]
    acts = list(results_dict.keys())
    final_vars = [np.mean(results_dict[act]['latent_var'][-1]) for act in acts]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(acts)]
    
    bars = ax4.bar([a.upper() for a in acts], final_vars, color=colors, alpha=0.85)
    ax4.set_title("Final Layer Latent Variance (Lower = Squeezed)")
    ax4.set_ylabel("Activation Variance (sigma^2)")
    ax4.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_dashboard.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Visualizer] Research dashboard saved successfully to: {save_path}")
