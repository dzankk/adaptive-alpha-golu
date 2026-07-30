# adaptive-alpha-golu
An adaptive extension of the GoLU activation function implementing learnable layer-by-layer asymmetry via backpropagation across CV and NLP benchmarks.

The canonical activation math lives in `models/alpha_golu.py`. The benchmark runners in `experiments/` now use real datasets for the main detection and language-model tasks, including Pascal VOC 2012 and WikiText-2.
