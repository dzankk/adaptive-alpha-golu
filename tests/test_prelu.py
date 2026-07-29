"""
PReLU Baseline Verification Test
================================
Tests that PReLU and Adaptive Alpha-GoLU execute cleanly via train_and_eval.
"""

import unittest
from experiments.run_classification import train_and_eval


class TestActivationRunners(unittest.TestCase):

    def test_classification_prelu(self):
        """Quick smoke test running 1 epoch on classification with prelu."""
        acc = train_and_eval(activation="prelu", seed=42, epochs=1)
        self.assertIsInstance(acc, float)
        self.assertGreater(acc, 0.0)

    def test_classification_alpha_golu(self):
        """Quick smoke test running 1 epoch on classification with alpha_golu."""
        acc = train_and_eval(activation="alpha_golu", seed=42, epochs=1)
        self.assertIsInstance(acc, float)
        self.assertGreater(acc, 0.0)


if __name__ == "__main__":
    unittest.main()
