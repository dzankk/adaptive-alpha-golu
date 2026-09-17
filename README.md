# Adaptive Alpha-GoLU: A Learnable Per-Layer Generalization of the Gompertz Linear Unit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Adaptive Alpha-GoLU** is a per-layer learnable generalization of the **Gompertz Linear Unit (GoLU)** activation function. It introduces a learnable asymmetry parameter (α), optimized directly via backpropagation per layer (and per channel, depending on architecture), and empirically tests whether this improves accuracy, robustness, or training stability compared to fixed-shape GoLU — across six tasks spanning computer vision and language modeling.

---

## 📄 Paper

The full write-up — including statistical testing, an honest account of the mixed results, and the engineering bugs found and fixed along the way — is here:

- **[Report (PDF)](paper/adaptive_alpha_golu_report.pdf)**
- **[LaTeX source](paper/adaptive_alpha_golu_report.tex)**

**tl;dr of the findings:** Alpha-GoLU is statistically indistinguishable from static GoLU on 5 of 6 tasks, with one small but statistically significant improvement on language modeling (p = 0.018) and one clear regression on diffusion. It also adds a measurable latency overhead over static GoLU and over ReLU/GELU/Swish, mostly in the backward pass — **this is not a free improvement**. The report treats this as a mixed, honestly-reported result rather than a clean win, and puts equal weight on the reproducible experimental pipeline itself.

---

## Key Features

- **Learnable per-layer asymmetry (α):** a per-layer (or per-channel) scalar, optimized via backpropagation, initialized so training starts identical to static GoLU.
- **Full statistical testing, not just point estimates:** paired/Welch's t-tests, multi-seed evaluation (3+ seeds per condition), and dedicated overhead profiling.
- **Cross-domain benchmark suite:** Classification (CIFAR-10), Object Detection (Pascal VOC), Segmentation (Pascal VOC), Diffusion (CIFAR-10), Language Modeling, and Corruption Robustness (Gaussian noise, shot noise, blur on CIFAR-10).
- **Reproducible pipeline:** checkpoint/resume on interruption, and a CLI that regenerates every table and figure in the paper directly from saved run data.

> ⚠️ **Note on dataset naming:** the Language Modeling task is described in the paper as a "synthetic/token corpus." *(Double-check this against `experiments/run_language_model.py` if it's actually WikiText-2 — pick whichever is accurate and make sure the README and the paper agree.)*

---

## Repository Architecture

```text
adaptive-alpha-golu/
├── configs/          # Benchmark and hyperparameter configuration JSONs
├── diagnostics/      # Trajectory trackers for alpha dynamics and gradient stability
├── experiments/      # Task-specific runners (Classification, Detection, LM, etc.)
├── models/           # Core PyTorch modules (alpha_golu.py, backbones)
├── outputs/          # Execution runs, checkpoints, and exported paper assets
├── paper/            # Final report (PDF + LaTeX source)
├── tests/            # Unit testing suite for stability and layer equivalence
├── utils/            # Data loaders, LaTeX exporters, and plot generators
├── cli.py            # Main entrypoint for benchmarks and paper asset generation
└── paper_benchmark.json
```

---

## Citation & License

This project is licensed under the [MIT License](LICENSE.md).

If referencing this work, please cite the report in `paper/`, and see [`kopic2026critical`] for the earlier critical analysis of GoLU that motivated this project.
