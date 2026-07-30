"""
Unit tests for activation layer instantiation across all comparative baselines.
"""

import unittest
import tempfile
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
from models.baselines import get_activation_layer
from utils.experiment_config import load_benchmark_config
from utils.stats import calculate_p_value
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json


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

    def test_run_manifest_artifact_writer(self):
        """Verify run artifacts can be created and serialized."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = create_run_directory(tmp_dir, task="classification", activation="alpha_golu", seeds=[42, 123])
            manifest = build_run_manifest(
                command="run_all",
                task="classification",
                seeds=[42, 123],
                activations=["relu", "alpha_golu"],
                extra_config={"epochs": 2},
            )
            write_json(Path(run_dir) / "run_manifest.json", manifest)

            self.assertTrue((Path(run_dir) / "run_manifest.json").exists())
            self.assertEqual(manifest["task"], "classification")
            self.assertEqual(manifest["seeds"], [42, 123])
            self.assertIn("environment", manifest)

    def test_benchmark_config_loader(self):
        """Verify the checked-in paper benchmark config can be loaded."""
        config = load_benchmark_config("configs/paper_benchmark.json")
        self.assertEqual(config["name"], "paper_benchmark_suite")
        self.assertEqual(config["seeds"], [42, 123, 999, 2024, 2025])
        self.assertIn("classification", config["tasks"])
        self.assertIn("alpha_golu", config["activations"])


if __name__ == "__main__":
    unittest.main()
