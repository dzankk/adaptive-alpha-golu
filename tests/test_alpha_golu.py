"""
Unit tests for AlphaGoLU module mechanics, parameter constraints, and forward passes.
"""

import unittest
import torch
from models.alpha_golu import AlphaGoLU, StaticGoLU


class TestAlphaGoLU(unittest.TestCase):

    def test_alpha_positivity(self):
        """Verify that alpha stays strictly positive via softplus."""
        layer = AlphaGoLU(init_alpha=1.0)
        self.assertGreater(layer.get_alpha_val().item(), 0.0)

    def test_forward_shape_preservation(self):
        """Verify tensor shapes are preserved across 2D, 3D, and 4D inputs."""
        layer_scalar = AlphaGoLU(num_parameters=1)
        layer_channel = AlphaGoLU(num_parameters=64)

        x_2d = torch.randn(8, 64)
        x_3d = torch.randn(8, 16, 64)
        x_4d = torch.randn(8, 64, 32, 32)

        self.assertEqual(layer_scalar(x_2d).shape, x_2d.shape)
        self.assertEqual(layer_channel(x_4d).shape, x_4d.shape)
        self.assertEqual(layer_channel(x_3d).shape, x_3d.shape)

    def test_gradient_flow(self):
        """Verify backpropagation reaches raw_alpha parameter."""
        layer = AlphaGoLU(init_alpha=1.0)
        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x).sum()
        out.backward()

        self.assertIsNotNone(layer.raw_alpha.grad)
        self.assertFalse(torch.isnan(layer.raw_alpha.grad).any())


if __name__ == "__main__":
    unittest.main()
