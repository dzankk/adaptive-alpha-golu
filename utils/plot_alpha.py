import matplotlib.pyplot as plt

def save_alpha_plot(layer1_history, layer2_history, filename="alpha_trajectory.png"):
    epochs = range(1, len(layer1_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, layer1_history, label="Layer 1 Alpha (Feature Extraction)", marker='o')
    plt.plot(epochs, layer2_history, label="Layer 2 Alpha (High-Level Latent)", marker='s')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label="Static GoLU Baseline (1.0)")
    plt.xlabel("Epochs")
    plt.ylabel("Learned Alpha Value")
    plt.title("Alpha Parameter Trajectory Across Network Depths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved successfully as '{filename}'!")
