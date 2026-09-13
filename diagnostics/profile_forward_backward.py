"""
Forward/Backward CUDA Timing Verification
==========================================
Uses torch.profiler (not the hand-rolled time.perf_counter timer in
utils/overhead_tracker.py) to get an authoritative, kernel-level breakdown of
forward vs backward time for a given task/activation. torch.profiler's CUDA
tracing (Kineto/CUPTI) attributes actual GPU kernel completion time back to
the enclosing named region regardless of asynchronous CPU/GPU dispatch, so
this settles whether a "backward faster than forward" measurement reflects a
real property of the model or a timing artifact.

Usage:
    python -m diagnostics.profile_forward_backward --task detection --activation alpha_golu
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.profile_complexity import TASK_SPECS, build_model, build_sample, training_loss


def profile_forward_backward(task_name: str, activation: str, device: torch.device, warmup: int = 5, measured: int = 10) -> None:
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

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(measured):
            model.zero_grad(set_to_none=True)
            with record_function("forward_pass"):
                loss = training_loss(task_name, model, sample, lm_vocab_size)
            with record_function("backward_pass"):
                loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    key_averages = prof.key_averages()
    time_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    forward_us = sum(getattr(event, time_key, 0) for event in key_averages if event.key == "forward_pass")
    backward_us = sum(getattr(event, time_key, 0) for event in key_averages if event.key == "backward_pass")

    print(f"\n=== {task_name} / {activation} on {device} (n={measured} measured iters, {warmup} warmup) ===")
    print(f"Forward:  {forward_us / measured / 1000:.3f} ms/iter  ({time_key}, torch.profiler ground truth)")
    print(f"Backward: {backward_us / measured / 1000:.3f} ms/iter  ({time_key}, torch.profiler ground truth)")

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(f"\nTop ops by {sort_key}:")
    print(key_averages.table(sort_by=sort_key, row_limit=15))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify forward/backward timing using torch.profiler (ground truth, independent of the custom OverheadTracker timer)"
    )
    parser.add_argument("--task", type=str, default="detection", choices=list(TASK_SPECS.keys()))
    parser.add_argument("--activation", type=str, default="alpha_golu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_forward_backward(args.task, args.activation, device, warmup=args.warmup, measured=args.iters)


if __name__ == "__main__":
    main()
