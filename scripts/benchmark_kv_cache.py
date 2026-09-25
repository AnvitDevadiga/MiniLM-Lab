import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from minilm_lab.generation import generate
from minilm_lab.kv_cache import generate_with_kv_cache
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_checkpoint.pt"))
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 3:
        parser.error("tokens must be positive and repeats must be at least three")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MiniLM(MiniLMConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = ByteTokenizer() if checkpoint.get("tokenizer", "byte") == "byte" else BPETokenizer.load(checkpoint["tokenizer"])
    prompt = "The experiment shows "
    modes = {
        "uncached": lambda: generate(model, prompt, tokenizer, args.tokens, top_k=None),
        "cached": lambda: generate_with_kv_cache(model, prompt, tokenizer, args.tokens),
    }
    timings = {name: [] for name in modes}
    for name, operation in modes.items():
        for repeat in range(args.repeats + 1):
            torch.manual_seed(1234)
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            operation()
            if device == "mps":
                torch.mps.synchronize()
            if repeat:
                timings[name].append(time.perf_counter() - start)
    uncached = statistics.median(timings["uncached"])
    cached = statistics.median(timings["cached"])
    result = {
        "device": device,
        "generated_tokens": args.tokens,
        "repeats": args.repeats,
        "warmup_runs_per_mode": 1,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "uncached_seconds_samples": timings["uncached"],
        "cached_seconds_samples": timings["cached"],
        "uncached_seconds": uncached,
        "cached_seconds": cached,
        "uncached_tokens_per_second": args.tokens / uncached,
        "cached_tokens_per_second": args.tokens / cached,
        "speedup": uncached / cached,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
