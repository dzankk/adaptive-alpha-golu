"""
PReLU Baseline Benchmark Test
=============================
Evaluates Parametric ReLU (PReLU) as an adaptive activation baseline across multiple random seeds
using the classification runner.
"""

from experiments.run_classification import train_and_eval


def run_prelu_tests():
    print("================ Evaluating PReLU (Adaptive Baseline) ================")
    seeds = [42, 123, 999]
    results = []

    for seed in seeds:
        acc = train_and_eval("prelu", seed=seed)
        results.append(acc)
        print(f"PReLU Seed {seed} -> Accuracy: {acc:.2f}%")

    avg_acc = sum(results) / len(results)
    print(f"\nMean PReLU Accuracy: {avg_acc:.2f}%")


if __name__ == "__main__":
    run_prelu_tests()
