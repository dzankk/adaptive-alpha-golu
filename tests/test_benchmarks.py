"""
Unit tests for activation layer instantiation across all comparative baselines.
"""

import unittest
import tempfile
import math
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
from models.alpha_golu import AlphaGoLU
from models.baselines import get_activation_layer
from experiments.run_segmentation import _resolve_segmentation_alpha_lr
from utils.experiment_config import load_benchmark_config
from utils.stats import (
    calculate_cohens_d,
    calculate_p_value,
    calculate_paired_cohens_d,
    format_timing_table,
    summarize_timing_by_activation,
)
from utils.scaled_benchmark_logger import build_significance_report, load_results_by_activation
from utils.run_artifacts import build_run_manifest, create_run_directory, stable_seed_directory, write_json
from utils.train_tuning import find_latest_checkpoint, load_training_checkpoint, save_training_checkpoint, set_activation_parameters_trainable, clear_training_checkpoints


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
        self.assertEqual(config["seeds"], [42, 123, 999])
        self.assertIn("classification", config["tasks"])
        self.assertIn("alpha_golu", config["activations"])

    def test_segmentation_alpha_lr_resolves_from_config(self):
        """Segmentation should honor the task alpha_lr config unless the caller overrides it explicitly."""
        resolved_default = _resolve_segmentation_alpha_lr(
            None,
            base_lr=0.02,
            alpha_lr_multiplier=10.0,
            config_path="configs/paper_benchmark.json",
        )
        self.assertAlmostEqual(resolved_default, 2e-4)

        resolved_override = _resolve_segmentation_alpha_lr(
            1e-3,
            base_lr=0.02,
            alpha_lr_multiplier=10.0,
            config_path="configs/paper_benchmark.json",
        )
        self.assertAlmostEqual(resolved_override, 1e-3)

    def test_activation_params_can_be_unfrozen_after_freeze(self):
        """Regression test: a freeze->unfreeze cycle must actually re-enable requires_grad.

        Previously, set_activation_parameters_trainable(model, True) silently did nothing once
        params had already been frozen, because it filtered on the very requires_grad flag it
        was supposed to set. This caused AlphaGoLU's alpha to stay permanently frozen after any
        warmup period, making it mathematically identical to StaticGoLU.
        """
        model = nn.Sequential(AlphaGoLU(init_alpha=1.0))

        set_activation_parameters_trainable(model, False)
        self.assertFalse(model[0].raw_alpha.requires_grad)

        enabled_count = set_activation_parameters_trainable(model, True)
        self.assertEqual(enabled_count, 1)
        self.assertTrue(model[0].raw_alpha.requires_grad)

        x = torch.randn(4, 8, requires_grad=False)
        model[0](x).sum().backward()
        self.assertIsNotNone(model[0].raw_alpha.grad)
        self.assertNotEqual(float(model[0].raw_alpha.grad.abs().sum()), 0.0)

    def test_checkpoint_round_trip_resumes_from_next_epoch(self):
        """An epoch-3 checkpoint should restore model/optimizer state and report epoch 3 for resume."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_dir = stable_seed_directory(tmp_dir, "detection", "alpha_golu", 42)

            model = nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

            model(torch.randn(3, 4)).sum().backward()
            optimizer.step()
            scheduler.step()
            save_training_checkpoint(
                seed_dir, 3, model=model, optimizer=optimizer, scheduler=scheduler,
                extra={"epoch_losses": [1.0, 0.8, 0.6]},
            )
            self.assertEqual(find_latest_checkpoint(seed_dir).name, "checkpoint_epoch_3.pth")

            # A later epoch's checkpoint must replace, not accumulate alongside, the earlier one.
            model(torch.randn(3, 4)).sum().backward()
            optimizer.step()
            scheduler.step()
            save_training_checkpoint(
                seed_dir, 4, model=model, optimizer=optimizer, scheduler=scheduler,
                extra={"epoch_losses": [1.0, 0.8, 0.6, 0.5]},
            )
            remaining = sorted(seed_dir.glob("checkpoint_epoch_*.pth"))
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].name, "checkpoint_epoch_4.pth")

            fresh_model = nn.Linear(4, 2)
            fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
            fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1)
            checkpoint = load_training_checkpoint(seed_dir)
            fresh_model.load_state_dict(checkpoint["model_state"])
            fresh_optimizer.load_state_dict(checkpoint["optimizer_state"])
            fresh_scheduler.load_state_dict(checkpoint["scheduler_state"])

            self.assertEqual(checkpoint["epoch"], 4)
            self.assertEqual(checkpoint["extra"]["epoch_losses"], [1.0, 0.8, 0.6, 0.5])
            self.assertTrue(torch.equal(fresh_model.weight, model.weight))

            clear_training_checkpoints(seed_dir)
            self.assertIsNone(find_latest_checkpoint(seed_dir))
            self.assertFalse(seed_dir.exists())

    def test_paired_p_value_uses_ttest_rel_for_matching_seed_lists(self):
        """Same-length score lists (matching seeds) should use the paired test by default."""
        # Same absolute noise every seed -> zero variance in the *difference*, so the paired
        # test detects a perfectly consistent improvement that Welch's test (unpaired) would not.
        baseline = [90.0, 80.0, 70.0]
        proposed = [90.5, 80.5, 70.5]

        p_paired = calculate_p_value(baseline, proposed, paired=True)
        p_welch = calculate_p_value(baseline, proposed, paired=False)

        self.assertLess(p_paired, 0.01)
        self.assertGreater(p_welch, p_paired)

    def test_paired_p_value_falls_back_to_welch_on_length_mismatch(self):
        """Unequal-length lists can't be paired by seed, so this must fall back to Welch's test."""
        baseline = [90.0, 80.0, 70.0]
        proposed = [91.0, 81.0]

        p_paired_requested = calculate_p_value(baseline, proposed, paired=True)
        p_welch_explicit = calculate_p_value(baseline, proposed, paired=False)
        self.assertAlmostEqual(p_paired_requested, p_welch_explicit)

    def test_cohens_d_effect_sizes(self):
        """Cohen's d (independent) and d_z (paired) should agree in sign for a consistent shift."""
        baseline = [90.0, 80.0, 70.0]
        proposed = [91.0, 81.0, 71.0]

        d = calculate_cohens_d(baseline, proposed)
        d_paired = calculate_paired_cohens_d(baseline, proposed)
        self.assertGreater(d, 0.0)
        self.assertTrue(math.isnan(d_paired))  # zero-variance differences -> undefined d_z

    def test_timing_table_reports_ratios_relative_to_baseline(self):
        """summarize_timing_by_activation should report activation timings relative to golu_static."""
        results_by_activation = {
            "golu_static": [{"epoch_seconds": [10.0, 10.0], "train_seconds": 20.0}],
            "alpha_golu": [{"epoch_seconds": [12.0, 12.0], "train_seconds": 24.0}],
        }
        timing = summarize_timing_by_activation(results_by_activation, baseline_activation="golu_static")

        self.assertAlmostEqual(timing["golu_static"]["mean_epoch_seconds"], 10.0)
        self.assertAlmostEqual(timing["alpha_golu"]["mean_epoch_seconds"], 12.0)
        self.assertAlmostEqual(timing["alpha_golu"]["epoch_seconds_vs_baseline"], 1.2)
        self.assertAlmostEqual(timing["golu_static"]["epoch_seconds_vs_baseline"], 1.0)

        table = format_timing_table(timing, baseline_activation="golu_static")
        self.assertIn("alpha_golu", table)
        self.assertIn("golu_static", table)

    def test_scaled_benchmark_logger_reads_run_artifacts_and_computes_significance(self):
        """build_significance_report should load results.json files by activation and compute stats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "language_model"
            for activation, scores in (("golu_static", [100.0, 102.0, 101.0]), ("alpha_golu", [98.0, 99.0, 97.0])):
                for seed, score in zip([42, 123, 999], scores):
                    run_dir = run_root / f"20260101T000000Z_language_model_{activation}_seeds-{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    write_json(
                        run_dir / "results.json",
                        {
                            "activation": activation,
                            "perplexity": score,
                            "epoch_seconds": [5.0, 5.0],
                            "train_seconds": 10.0,
                        },
                    )

            loaded = load_results_by_activation("language_model", ["golu_static", "alpha_golu"], output_root=tmp_dir)
            self.assertEqual(len(loaded["golu_static"]), 3)
            self.assertEqual(len(loaded["alpha_golu"]), 3)

            report = build_significance_report(
                ["language_model"], ["golu_static", "alpha_golu"], output_root=tmp_dir
            )
            task_report = report["tasks"]["language_model"]
            self.assertEqual(task_report["baseline_scores"], [100.0, 102.0, 101.0])
            self.assertEqual(task_report["proposed_scores"], [98.0, 99.0, 97.0])
            self.assertIsNotNone(task_report["p_value_paired"])
            self.assertIsNotNone(task_report["cohens_d"])
            self.assertEqual(task_report["timing"]["golu_static"]["mean_train_seconds"], 10.0)

    def test_phase2_scale_up_config_loads(self):
        """The Phase 2 scaffold config should be valid and loadable like other benchmark configs."""
        config = load_benchmark_config("configs/phase2_complex_scale.json")
        self.assertEqual(config["name"], "phase2_complex_scale")
        self.assertIn("classification", config["tasks"])
        self.assertIn("classification_scale_up", config)
        self.assertIn("language_model_scale_up", config)
        self.assertIn("diffusion_scale_up", config)


if __name__ == "__main__":
    unittest.main()
