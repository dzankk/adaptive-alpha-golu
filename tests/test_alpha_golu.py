"""
Unit tests for AlphaGoLU module mechanics, parameter constraints, and forward passes.
"""

import unittest
import math
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
        layer_channels_kw = AlphaGoLU(channels=64)

        x_2d = torch.randn(8, 64)
        x_3d = torch.randn(8, 16, 64)
        x_4d = torch.randn(8, 64, 32, 32)

        self.assertEqual(layer_scalar(x_2d).shape, x_2d.shape)
        self.assertEqual(layer_channel(x_4d).shape, x_4d.shape)
        self.assertEqual(layer_channel(x_3d).shape, x_3d.shape)
        self.assertEqual(layer_channels_kw(x_4d).shape, x_4d.shape)
        self.assertEqual(tuple(layer_channels_kw.raw_alpha.shape), (1, 64, 1, 1))

    def test_gradient_flow(self):
        """Verify backpropagation reaches raw_alpha parameter."""
        layer = AlphaGoLU(init_alpha=1.0)
        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x).sum()
        out.backward()

        self.assertIsNotNone(layer.raw_alpha.grad)
        self.assertFalse(torch.isnan(layer.raw_alpha.grad).any())

    def test_alpha_clamping(self):
        """Verify alpha can be hard-clamped to a safe interval."""
        layer = AlphaGoLU(channels=4, init_alpha=10.0)
        layer.clamp_alpha_(0.2, 3.0)
        self.assertGreaterEqual(layer.get_alpha_val().min().item(), 0.2)
        self.assertLessEqual(layer.get_alpha_val().max().item(), 3.0)

    def test_init_alpha_round_trip(self):
        """Verify init_alpha maps back to alpha≈1.0 through inverse softplus."""
        layer = AlphaGoLU(init_alpha=1.0)
        self.assertTrue(torch.isclose(layer.alpha, torch.tensor(1.0), atol=1e-4, rtol=1e-4).item())

    def test_static_activation_matches_formula(self):
        """Verify the static gate matches x * exp(-exp(-x))."""
        layer = StaticGoLU()
        x = torch.tensor([-2.0, 0.0, 2.0])
        expected = x * torch.exp(-torch.exp(-x))
        self.assertTrue(torch.allclose(layer(x), expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
