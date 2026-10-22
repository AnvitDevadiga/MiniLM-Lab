"""Matched training-step throughput and memory for each attention backend.

One step = forward + cross-entropy + backward + AdamW update on a batch of
random tokens. All backends share the same seeded initial weights and the
same batches; each backend has its own model copy and optimizer so the
weights stay comparable. Backend order rotates every repeat. Peak memory is
measured on the first step in a fresh process per configuration (CPU: peak
RSS delta; MPS: live allocator bytes sampled after every op, including
backward); both include AdamW state allocation, which is identical across backends.
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.provenance import (
    environment,
    peak_rss_bytes,
    summarize,
    synchronize,
    write_record,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = {"vocab_size": 256, "embedding_dim": 192, "num_layers": 4, "num_heads": 6}


def csv(kind):
    return lambda text: [kind(item) for item in text.split(",")]


def build(args, length: int) -> MiniLM:
    torch.manual_seed(args.seed)
    return MiniLM(MiniLMConfig(context_length=length, **ARCHITECTURE)).to(args.device).train()


def batch_for(args, length: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(args.seed + 7919 * index + length)
    tokens = torch.randint(0, ARCHITECTURE["vocab_size"], (args.batch_size, length + 1), generator=generator).to(args.device)
    return tokens[:, :-1], tokens[:, 1:]


def step(model: MiniLM, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


class PeakAllocated(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.peak = torch.mps.current_allocated_memory()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        return output


def memory_probe(args) -> None:
    length = args.seq_lens[0]
    model = build(args, length).set_attention_backend(args.backends[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # The first step is measured: a warm-up step would already set the process
    # RSS high-water mark. It includes AdamW state allocation (same for all backends).
    inputs, targets = batch_for(args, length, 0)
    synchronize(args.device)
    if args.device == "mps":
        before = torch.mps.current_allocated_memory()
        with PeakAllocated() as tracker:
            step(model, optimizer, inputs, targets)
        synchronize(args.device)
        after, method = tracker.peak, "mps_live_allocated_peak_per_op"
    else:
        before = peak_rss_bytes()
        step(model, optimizer, inputs, targets)
        after, method = peak_rss_bytes(), "process_peak_rss_delta"
    print(json.dumps({"peak_extra_bytes": after - before, "method": method}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--backends", type=csv(str), default=["mha", "sdpa", "reference"])
    parser.add_argument("--seq-lens", type=csv(int), default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/training_step")
    args = parser.parse_args()
    if args.memory_probe:
        memory_probe(args)
        return
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment")
    results = []
    for length in args.seq_lens:
        base = build(args, length)
        models = {backend: copy.deepcopy(base).set_attention_backend(backend) for backend in args.backends}
        optimizers = {backend: torch.optim.AdamW(model.parameters(), lr=3e-4) for backend, model in models.items()}
        samples = {backend: [] for backend in args.backends}
        losses = {backend: [] for backend in args.backends}
        for repeat in range(args.warmup + args.repeats):
            inputs, targets = batch_for(args, length, repeat)
            shift = repeat % len(args.backends)
            for backend in args.backends[shift:] + args.backends[:shift]:
                synchronize(args.device)
                start = time.perf_counter()
                loss = step(models[backend], optimizers[backend], inputs, targets)
                synchronize(args.device)
                elapsed = time.perf_counter() - start
                losses[backend].append(loss)
                if repeat >= args.warmup:
                    samples[backend].append(elapsed)
        for backend in args.backends:
            command = [sys.executable, __file__, "--memory-probe", "--device", args.device, "--backends", backend,
                       "--seq-lens", str(length), "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
            memory = json.loads(subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip().splitlines()[-1])
            tokens = args.batch_size * length
            row = {
                "backend": backend, "seq_len": length, "batch": args.batch_size,
                "step_ms": summarize([1000 * s for s in samples[backend]]),
                "tokens_per_second": summarize([tokens / s for s in samples[backend]]),
                "step_seconds_samples": samples[backend],
                "losses": losses[backend],
                "max_abs_loss_difference_vs_first_backend": max(abs(a - b) for a, b in zip(losses[backend], losses[args.backends[0]])),
                "memory": memory,
            }
            results.append(row)
            print(f"{args.device} L={length:5d} {backend:9s} step {row['step_ms']['median']:8.1f} ms  "
                  f"{row['tokens_per_second']['median']:8.0f} tok/s  mem {memory['peak_extra_bytes'] / 2**20:8.1f} MiB  "
                  f"max|Δloss| {row['max_abs_loss_difference_vs_first_backend']:.2e}", flush=True)
    payload = {
        "experiment": "training_step",
        "environment": environment(args.device),
        "settings": {"dtype": "float32", "backends": args.backends, "seq_lens": args.seq_lens, "batch_size": args.batch_size,
                     "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed, "optimizer": "AdamW lr=3e-4",
                     "architecture": ARCHITECTURE, "weights": "seeded random initialization, identical across backends"},
        "results": results,
    }
    print(f"wrote {write_record(args.output_dir, f'training_step_{args.device}_float32', payload)}")


if __name__ == "__main__":
    main()
