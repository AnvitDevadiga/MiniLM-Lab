"""Cached vs uncached autoregressive decoding: latency, equivalence and memory growth.

Modes (identical weights, inputs, dtype and device):
  cached           KV cache, SDPA attention (what forward_cached always uses)
  uncached_sdpa    full recompute each step, SDPA attention
  uncached_default full recompute each step, the repository default attention
The two uncached modes separate the cache effect from the attention kernel.

Every step (step 0 = prompt prefill) is synchronized and timed. The timed loop
is checked to emit exactly the tokens of minilm_lab.kv_cache.greedy_decode.

Scenarios:
  grid    prompt lengths x batch sizes, all inside the context window
  window  generation far past the context window (sliding refresh)
  memory  cache bytes per step and, on MPS, live allocator peak per step
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from minilm_lab.kv_cache import greedy_decode
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.provenance import environment, summarize, synchronize, write_record

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
ARCHITECTURE = {"vocab_size": 256, "embedding_dim": 192, "num_layers": 4, "num_heads": 6}
MODES = {"cached": ("sdpa", True), "uncached_sdpa": ("sdpa", False), "uncached_default": ("auto", False)}


def csv(kind):
    return lambda text: [kind(item) for item in text.split(",")]


def build(args, context_length: int) -> MiniLM:
    torch.manual_seed(args.seed)
    config = MiniLMConfig(context_length=context_length, positional_encoding=args.position, **ARCHITECTURE)
    return MiniLM(config).to(device=args.device, dtype=DTYPES[args.dtype]).eval()


def prompt_for(args, batch: int, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(args.seed + 1000 * batch + length)
    return torch.randint(0, ARCHITECTURE["vocab_size"], (batch, length), generator=generator).to(args.device)


@torch.inference_mode()
def timed_decode(model: MiniLM, prompt: torch.Tensor, new_tokens: int, use_cache: bool, device: str, on_step=None):
    """greedy_decode with per-step timing; on_step(index, cache) may inspect state."""
    window = model.config.context_length
    tokens, cache, times, refreshed = prompt, None, [], []
    for index in range(new_tokens):
        synchronize(device)
        start = time.perf_counter()
        if not use_cache:
            logits, _ = model(tokens[:, -window:])
            refresh = True
        elif cache is None or tokens.size(1) > window:
            logits, cache = model.forward_cached(tokens[:, -window:])
            refresh = True
        else:
            logits, cache = model.forward_cached(tokens[:, -1:], cache)
            refresh = False
        tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        synchronize(device)
        times.append(time.perf_counter() - start)
        refreshed.append(refresh)
        if on_step:
            on_step(index, cache)
    return tokens, times, refreshed


def run_modes(args, model: MiniLM, prompt: torch.Tensor, new_tokens: int) -> dict:
    samples = {mode: [] for mode in MODES}
    outputs = {}
    names = list(MODES)
    for repeat in range(args.warmup + args.repeats):
        shift = repeat % len(names)
        for mode in names[shift:] + names[:shift]:
            backend, use_cache = MODES[mode]
            model.set_attention_backend(backend)
            tokens, times, refreshed = timed_decode(model, prompt, new_tokens, use_cache, args.device)
            outputs[mode] = tokens
            if repeat >= args.warmup:
                samples[mode].append({"step_seconds": times, "refreshed": refreshed})
    model.set_attention_backend("sdpa")
    reference = greedy_decode(model, prompt, new_tokens, use_cache=True)
    if not torch.equal(reference, outputs["cached"]):
        raise AssertionError("timed loop diverged from greedy_decode")
    batch = prompt.size(0)
    result = {"token_identical_to_cached": {mode: bool(torch.equal(outputs[mode], outputs["cached"])) for mode in MODES}}
    for mode, runs in samples.items():
        totals = [sum(run["step_seconds"]) for run in runs]
        result[mode] = {
            "first_token_ms": summarize([1000 * run["step_seconds"][0] for run in runs]),
            "decode_ms_per_step": summarize([1000 * t for run in runs for t in run["step_seconds"][1:]]),
            "total_ms": summarize([1000 * t for t in totals]),
            "tokens_per_second": summarize([batch * new_tokens / t for t in totals]),
            "median_step_ms_by_index": [
                1000 * sorted(run["step_seconds"][i] for run in runs)[len(runs) // 2] for i in range(new_tokens)
            ],
            "refreshed_steps": runs[0]["refreshed"],
        }
    result["speedup_cached_vs_uncached_sdpa"] = result["uncached_sdpa"]["total_ms"]["median"] / result["cached"]["total_ms"]["median"]
    result["speedup_cached_vs_uncached_default"] = result["uncached_default"]["total_ms"]["median"] / result["cached"]["total_ms"]["median"]
    return result


class PeakAllocated(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.peak = torch.mps.current_allocated_memory()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        return output


@torch.inference_mode()
def memory_growth(args, model: MiniLM, prompt: torch.Tensor, new_tokens: int) -> dict:
    model.set_attention_backend("sdpa")
    window = model.config.context_length
    tokens, cache, rows = prompt, None, []
    for index in range(new_tokens):
        tracker = PeakAllocated() if args.device == "mps" else None
        before = torch.mps.current_allocated_memory() if tracker else None
        if tracker:
            tracker.__enter__()
        if cache is None or tokens.size(1) > window:
            logits, cache = model.forward_cached(tokens[:, -window:])
        else:
            logits, cache = model.forward_cached(tokens[:, -1:], cache)
        if tracker:
            tracker.__exit__(None, None, None)
        tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        row = {
            "step": index,
            "cached_positions": cache[0][0].size(2),
            "cache_bytes": sum(t.numel() * t.element_size() for layer in cache for t in layer),
        }
        if tracker:
            row["step_peak_minus_start_bytes"] = tracker.peak - before
        rows.append(row)
    return {"batch": prompt.size(0), "prompt_len": prompt.size(1), "steps": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--position", choices=("learned", "rope"), default="learned")
    parser.add_argument("--prompt-lens", type=csv(int), default=[1, 16, 128, 512, 1024])
    parser.add_argument("--batch-sizes", type=csv(int), default=[1, 8])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--window-context", type=int, default=128, help="context for the sliding-window scenario")
    parser.add_argument("--window-prompt", type=int, default=32)
    parser.add_argument("--window-new-tokens", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/kv_cache")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment")
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least one warmup and three repeats")

    context = max(args.prompt_lens) + args.new_tokens
    model = build(args, context)
    grid = []
    for batch in args.batch_sizes:
        for length in args.prompt_lens:
            row = {"batch": batch, "prompt_len": length, **run_modes(args, model, prompt_for(args, batch, length), args.new_tokens)}
            grid.append(row)
            print(
                f"{args.device}/{args.dtype} B={batch} L={length:5d} cached {row['cached']['total_ms']['median']:9.1f} ms  "
                f"uncached_sdpa {row['uncached_sdpa']['total_ms']['median']:9.1f} ms  uncached_default {row['uncached_default']['total_ms']['median']:9.1f} ms  "
                f"identical={row['token_identical_to_cached']}",
                flush=True,
            )
    window_model = build(args, args.window_context)
    window = {"batch": 1, "prompt_len": args.window_prompt, "context_length": args.window_context, "new_tokens": args.window_new_tokens}
    window.update(run_modes(args, window_model, prompt_for(args, 1, args.window_prompt), args.window_new_tokens))
    print(f"window: cached {window['cached']['total_ms']['median']:.1f} ms vs uncached_sdpa {window['uncached_sdpa']['total_ms']['median']:.1f} ms", flush=True)
    memory_prompt = min(128, context - args.new_tokens)
    memory = [memory_growth(args, model, prompt_for(args, batch, memory_prompt), args.new_tokens) for batch in args.batch_sizes]
    memory.append(memory_growth(args, window_model, prompt_for(args, 1, args.window_prompt), args.window_new_tokens))
    payload = {
        "experiment": "kv_cache_scaling",
        "environment": environment(args.device),
        "settings": {
            "dtype": args.dtype,
            "position": args.position,
            "prompt_lens": args.prompt_lens,
            "batch_sizes": args.batch_sizes,
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "modes": {mode: {"attention_backend": b, "use_cache": c} for mode, (b, c) in MODES.items()},
            "weights": "seeded random initialization (timing does not depend on trained values)",
            "grid_model_config": model.config.__dict__,
            "window_model_config": window_model.config.__dict__,
            "decoding": "greedy argmax; step 0 is the prompt prefill",
        },
        "results": {"grid": grid, "window": window, "memory": memory},
    }
    print(f"wrote {write_record(args.output_dir, f'kv_{args.device}_{args.dtype}', payload)}")


if __name__ == "__main__":
    main()
