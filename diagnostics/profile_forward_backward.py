"""
Forward/Backward CUDA Timing Verification
==========================================
Times forward vs backward using CUDA events (or perf_counter+sync on CPU), and
separately uses torch.profiler purely to print an informational kernel-level
breakdown table. CUDA events are the ground truth here -- NOT
torch.profiler's self_device_time_total attributed to a record_function
marker: on CUDA, backward() can execute its kernels via the autograd engine
off the thread that opened the record_function scope, so the marker's own
"self" time comes back near-zero even though real backward work happened
(confirmed: this previously reported ~0.001 ms/iter backward time for every
activation/task on a real GPU run). CUDA events measure actual elapsed time
between two points on the stream regardless of which thread enqueued the
kernels in between, so they don't have this blind spot.

This also doubles as the FASTEST way to fully populate outputs/overhead/ for
the paper's runtime-overhead plot: unlike the OverheadTracker path (which only
records timing during a real training run, so a task only gets covered if you
happen to (re)run full training for every activation with tracking enabled),
this script profiles a handful of synthetic forward/backward passes per
(task, activation) in seconds -- looping over every activation for a task and
writing results into outputs/overhead/ takes minutes, not hours.

Usage:
    # Single (task, activation) -- just prints the timing + profiler breakdown:
    python -m diagnostics.profile_forward_backward --task detection --activation alpha_golu

    # All 8 canonical activations for a task, saved into outputs/overhead/:
    python -m diagnostics.profile_forward_backward --task detection --activations relu gelu swish prelu pgelu golu_static alpha_golu adaptive_swish --save
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.profile_complexity import ACTIVATIONS, TASK_SPECS, build_model, build_sample, training_loss
from utils.run_artifacts import write_json


def measure_forward_backward(
    task_name: str, activation: str, device: torch.device, warmup: int = 5, measured: int = 10, verbose: bool = True
) -> tuple[float, float]:
    """Runs forward/backward for one (task, activation) pair and returns (forward_ms, backward_ms)."""
    lm_vocab_size = 1000
    model = build_model(task_name, activation, device, lm_vocab_size)
    model.train()
    sample = build_sample(task_name, device, lm_vocab_size)

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        loss = training_loss(task_name, model, sample, lm_vocab_size)
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_times_ms: list[float] = []
    backward_times_ms: list[float] = []
    for _ in range(measured):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            start_fwd, end_fwd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_bwd, end_bwd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_fwd.record()
            loss = training_loss(task_name, model, sample, lm_vocab_size)
            end_fwd.record()
            start_bwd.record()
            loss.backward()
            end_bwd.record()
            torch.cuda.synchronize(device)
            forward_times_ms.append(start_fwd.elapsed_time(end_fwd))
            backward_times_ms.append(start_bwd.elapsed_time(end_bwd))
        else:
            t0 = time.perf_counter()
            loss = training_loss(task_name, model, sample, lm_vocab_size)
            t1 = time.perf_counter()
            loss.backward()
            t2 = time.perf_counter()
            forward_times_ms.append((t1 - t0) * 1000.0)
            backward_times_ms.append((t2 - t1) * 1000.0)

    forward_ms = sum(forward_times_ms) / measured
    backward_ms = sum(backward_times_ms) / measured

    if verbose:
        time_label = "CUDA events" if device.type == "cuda" else "perf_counter (CPU)"
        print(f"\n=== {task_name} / {activation} on {device} (n={measured} measured iters, {warmup} warmup) ===")
        print(f"Forward:  {forward_ms:.3f} ms/iter  ({time_label}, ground truth)")
        print(f"Backward: {backward_ms:.3f} ms/iter  ({time_label}, ground truth)")

        # Separate torch.profiler pass purely for the informational kernel-level breakdown table
        # below -- not used for the forward_ms/backward_ms numbers above (see module docstring).
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        table_iters = min(measured, 3)
        with profile(activities=activities, record_shapes=False) as prof:
            for _ in range(table_iters):
                model.zero_grad(set_to_none=True)
                with record_function("forward_pass"):
                    loss = training_loss(task_name, model, sample, lm_vocab_size)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                with record_function("backward_pass"):
                    loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(f"\nTop ops by {sort_key} (informational, {table_iters} iters -- not the timing source above):")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    return forward_ms, backward_ms



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify forward/backward timing using torch.profiler (ground truth, independent of the custom OverheadTracker timer)"
    )
    parser.add_argument("--task", type=str, default=None, choices=list(TASK_SPECS.keys()), help="Single task to profile (default: use --tasks instead)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASK_SPECS.keys()), help="Profile every one of these tasks in a single run (default: just --task, or detection if neither is given)")
    parser.add_argument("--activation", type=str, default=None, help="Single activation to profile (default: use --activations instead)")
    parser.add_argument("--activations", type=str, nargs="+", default=None, choices=ACTIVATIONS, help="Profile every one of these activations for each task in a single run (default: just --activation, or alpha_golu if neither is given)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="Write each (task, activation) result into outputs/overhead/ in the schema plot_paper_overhead_summary expects")
    parser.add_argument("--overhead-root", type=str, default="outputs/overhead", help="Directory to write results into when --save is set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = args.tasks or [args.task or "detection"]
    activations = args.activations or [args.activation or "alpha_golu"]
    verbose = len(tasks) == 1 and len(activations) == 1

    for task in tasks:
        results = []
        for activation in activations:
            forward_ms, backward_ms = measure_forward_backward(
                task, activation, device, warmup=args.warmup, measured=args.iters, verbose=verbose
            )
            results.append((activation, forward_ms, backward_ms))
            if args.save:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                overhead_root = Path(args.overhead_root)
                write_json(
                    overhead_root / f"{timestamp}_{task}_{activation}_profiler.json",
                    {
                        "task_name": task,
                        "activation_name": activation,
                        "forward_ms": {"mean": forward_ms},
                        "backward_ms": {"mean": backward_ms},
                        "source": "diagnostics.profile_forward_backward (CUDA events / perf_counter ground truth)",
                    },
                )

        if not verbose:
            print(f"\n=== Summary: {task} on {device} ===")
            print(f"{'Activation':<16} {'Forward (ms)':>14} {'Backward (ms)':>15}")
            for activation, forward_ms, backward_ms in results:
                print(f"{activation:<16} {forward_ms:>14.3f} {backward_ms:>15.3f}")
            if args.save:
                print(f"[IO] Saved {len(results)} record(s) to {args.overhead_root}")


if __name__ == "__main__":
    main()
