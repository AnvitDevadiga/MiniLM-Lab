"""Matched latency, throughput and memory comparison of attention backends.

Every backend runs on the same model object (identical seeded weights), the
same token inputs, dtype and device. Backend order rotates every repeat so
slow thermal or frequency drift does not favour one backend.

Phases per (batch, sequence length):
  forward  full-sequence logits via model(x): the training/evaluation path
  prefill  model.forward_cached(prompt): logits plus a KV cache
  decode   greedy single-token steps against the cache, each step timed

Peak memory is measured in a fresh subprocess per configuration because the
process high-water mark (CPU) and allocator pool (MPS) never shrink.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.provenance import (
    environment,
    peak_rss_bytes,
    summarize,
    synchronize,
    write_record,
)

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
# The architecture of the published science checkpoint. Only the context
# length (the position table) grows so that long sequences can be measured.
ARCHITECTURE = {"vocab_size": 256, "embedding_dim": 192, "num_layers": 4, "num_heads": 6}


def csv(kind):
    return lambda text: [kind(item) for item in text.split(",")]


def build(args, context_length: int) -> MiniLM:
    torch.manual_seed(args.seed)
    config = MiniLMConfig(context_length=context_length, positional_encoding=args.position, **ARCHITECTURE)
    return MiniLM(config).to(device=args.device, dtype=DTYPES[args.dtype]).eval()


def inputs(args, batch: int, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(args.seed + 1000 * batch + length)
    return torch.randint(0, ARCHITECTURE["vocab_size"], (batch, length), generator=generator).to(args.device)


def timed(device: str, operation):
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return time.perf_counter() - start, result


@torch.inference_mode()
def trial(model: MiniLM, prompt: torch.Tensor, decode_tokens: int, device: str, phases: set[str]) -> dict:
    sample = {}
    if "forward" in phases:
        sample["forward"], _ = timed(device, lambda: model(prompt))
    if model.blocks[0].attention.backend == "auto":
        # forward_cached already uses SDPA under "auto"; no distinct decode path.
        return sample
    sample["prefill"], (logits, cache) = timed(device, lambda: model.forward_cached(prompt))
    steps = []
    for _ in range(decode_tokens):
        token = logits[:, -1:].argmax(dim=-1)
        elapsed, (logits, cache) = timed(device, lambda token=token, cache=cache: model.forward_cached(token, cache))
        steps.append(elapsed)
    sample["decode_steps"] = steps
    return sample


@torch.inference_mode()
def memory_probe(args) -> None:
    """Run one forward pass in this fresh process and print peak extra memory."""
    model = build(args, args.context_length).set_attention_backend(args.backends[0])
    prompt = inputs(args, args.batch_sizes[0], args.seq_lens[0])
    synchronize(args.device)
    if args.device == "mps":
        torch.mps.empty_cache()
        before = torch.mps.driver_allocated_memory()
    else:
        before = peak_rss_bytes()
    model(prompt)
    synchronize(args.device)
    if args.device == "mps":
        # Freed blocks stay in the MPS pool, so this approximates the pool's
        # high-water mark during the pass (not an exact allocation peak).
        after, method = torch.mps.driver_allocated_memory(), "mps_driver_allocated_delta"
    else:
        after, method = peak_rss_bytes(), "process_peak_rss_delta"
    print(json.dumps({"peak_extra_bytes": after - before, "method": method}))


def measure_memory(args, backend: str, batch: int, length: int) -> dict:
    command = [
        sys.executable, __file__, "--memory-probe", "--device", args.device, "--dtype", args.dtype,
        "--position", args.position, "--seed", str(args.seed), "--backends", backend,
        "--batch-sizes", str(batch), "--seq-lens", str(length), "--context-length", str(args.context_length),
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    return json.loads(output.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--position", choices=("learned", "rope"), default="learned")
    parser.add_argument("--backends", type=csv(str), default=["auto", "sdpa", "reference"])
    parser.add_argument("--seq-lens", type=csv(int), default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=csv(int), default=[1, 4, 8])
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=None, help="defaults to max seq len + decode tokens")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--memory-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/attention")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.context_length is None:
        args.context_length = max(args.seq_lens) + args.decode_tokens
    if args.memory_probe:
        memory_probe(args)
        return
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least one warmup and three repeats")

    model = build(args, args.context_length)
    results = []
    for batch in args.batch_sizes:
        for length in args.seq_lens:
            prompt = inputs(args, batch, length)
            phases = {"forward"}
            samples = {backend: [] for backend in args.backends}
            for repeat in range(args.warmup + args.repeats):
                shift = repeat % len(args.backends)
                for backend in args.backends[shift:] + args.backends[:shift]:
                    model.set_attention_backend(backend)
                    sample = trial(model, prompt, args.decode_tokens, args.device, phases)
                    if repeat >= args.warmup:
                        samples[backend].append(sample)
            for backend, runs in samples.items():
                row = {"backend": backend, "batch": batch, "seq_len": length, "samples_seconds": runs}
                row["forward_ms"] = summarize([1000 * run["forward"] for run in runs])
                if "prefill" in runs[0]:
                    prefill = [run["prefill"] for run in runs]
                    decode_totals = [sum(run["decode_steps"]) for run in runs]
                    row["prefill_ms"] = summarize([1000 * value for value in prefill])
                    row["decode_ms_per_step"] = summarize([1000 * step for run in runs for step in run["decode_steps"]])
                    row["decode_tokens_per_second"] = summarize([batch * args.decode_tokens / total for total in decode_totals])
                    row["generation_tokens_per_second"] = summarize(
                        [batch * args.decode_tokens / (p + d) for p, d in zip(prefill, decode_totals)]
                    )
                if not args.skip_memory:
                    row["memory"] = measure_memory(args, backend, batch, length)
                results.append(row)
                print(
                    f"{args.device}/{args.dtype} B={batch} L={length:5d} {backend:9s} "
                    f"forward {row['forward_ms']['median']:8.2f} ms"
                    + (f"  prefill {row['prefill_ms']['median']:8.2f} ms  decode {row['decode_ms_per_step']['median']:6.2f} ms/step" if "prefill_ms" in row else "")
                    + (f"  mem {row['memory']['peak_extra_bytes'] / 2**20:8.1f} MiB" if "memory" in row else ""),
                    flush=True,
                )
    payload = {
        "experiment": "attention_backends",
        "environment": environment(args.device),
        "settings": {
            "dtype": args.dtype,
            "position": args.position,
            "backends": args.backends,
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "weights": "seeded random initialization (timing does not depend on trained values)",
            "model_config": model.config.__dict__,
            "inference_mode": True,
        },
        "results": results,
    }
    name = args.name or f"attention_{args.device}_{args.dtype}"
    print(f"wrote {write_record(args.output_dir, name, payload)}")


if __name__ == "__main__":
    main()
