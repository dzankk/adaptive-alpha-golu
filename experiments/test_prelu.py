
import torch
import torch.nn as nn
from models.baselines import DeepConvNet
from experiments.run_baselines import train_and_eval

print("================ Evaluating PReLU (Adaptive Baseline) ================")
for seed in [42, 123, 999]:
    acc = train_and_eval("prelu", seed=seed)
    print(f"PReLU Seed {seed} -> Accuracy: {acc:.2f}%")
