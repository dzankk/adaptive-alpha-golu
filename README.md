# Adaptive Alpha-GoLU: Learnable Asymmetrical Activation Functions for Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Adaptive Alpha-GoLU** is an adaptive extension of the Gaussian-Omitted Linear Unit (GoLU) activation function. It introduces learnable layer-wise (and channel-wise) asymmetry ($\alpha$ parameter) optimized directly via backpropagation to dynamically sculpt non-linearities across Computer Vision and Natural Language Processing architectures.

---

##  Key Features

- **Dynamic Asymmetry ($\alpha$ parameter):** Learnable scaling factors per layer or channel that adapt to gradient flow during training.
- **Zero Incomplete-Beta Bottleneck:** Optimized numerical formulation ensuring high throughput and negligible GPU overhead vs. standard GELU/ReLU.
- **Cross-Domain Benchmark Suite:** Built-in evaluation covering Classification, Object Detection (Pascal VOC), Image Segmentation, Generative Diffusion (UNet), Language Modeling (WikiText-2), and Adversarial Robustness.
- **Automated Paper Pipeline:** Reproducible CLI toolchain that generates LaTeX tables and figures directly from experiment artifacts.

---

##  Repository Architecture

```text
adaptive-alpha-golu/
├── configs/          # Benchmark and hyperparameter configuration JSONs
├── diagnostics/      # Trajectory trackers for alpha dynamics and gradient stability
├── experiments/      # Task-specific runners (Classification, Detection, LM, etc.)
├── models/           # Core PyTorch modules (alpha_golu.py, backbones)
├── outputs/          # Execution runs, checkpoints, and export paper assets
├── tests/            # Unit testing suite for stability and layer equivalence
├── utils/            # Data loaders, LaTeX exporters, and plot generators
├── cli.py            # Main entrypoint for benchmarks and paper asset generation
└── paper_benchmark.json
```


##  Citation & License

This project is licensed under the [MIT License](LICENSE.md).
