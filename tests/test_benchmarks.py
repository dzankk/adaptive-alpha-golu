"""
Unit tests for activation factory function across all comparative baselines.
"""

import unittest
import torch
import torch.nn as nn
from models.benchmarks import get_activation_layer


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


if __name__ == "__main__":
    unittest.main()
