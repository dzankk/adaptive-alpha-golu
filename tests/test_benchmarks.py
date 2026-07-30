"""
Unit tests for activation layer instantiation across all comparative baselines.
"""

import unittest
import torch
import torch.nn as nn
from models.baselines import get_activation_layer
from utils.stats import calculate_p_value


class TestActivationFactory(unittest.TestCase):

    def test_supported_activations(self):
        """Verify all baseline and proposed activation factory routes instantiate valid Modules."""
        activations = ['gelu', 'swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']
        x = torch.randn(2, 10)

        for act_name in activations:
            layer = get_activation_layer(act_name)
            self.assertIsInstance(layer, nn.Module)
            out = layer(x)
            self.assertEqual(out.shape, x.shape)

    def test_welch_p_value_smoke(self):
        """Verify the shared stats helper returns a valid Welch p-value."""
        p_val = calculate_p_value([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)


if __name__ == "__main__":
    unittest.main()
