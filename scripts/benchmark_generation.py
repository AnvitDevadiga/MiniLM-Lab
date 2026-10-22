import argparse
import json
import time
from pathlib import Path

import torch

from minilm_lab.generation import generate
from minilm_lab.model import (
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    MiniLM,
    MiniLMConfig,
)
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_checkpoint.pt"))
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--attention-backend", choices=(*ATTENTION_BACKENDS, "auto"), default=DEFAULT_ATTENTION_BACKEND,
        help="sdpa (default); mha reproduces published science-v1 measurements",
    )
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MiniLM(MiniLMConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_attention_backend(args.attention_backend)
    tokenizer = ByteTokenizer() if checkpoint.get("tokenizer", "byte") == "byte" else BPETokenizer.load(checkpoint["tokenizer"])
    start = time.perf_counter()
    generate(model, "To be or not to be", tokenizer, args.tokens)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    result = {"device": device, "attention_backend": model.blocks[0].attention.backend, "generated_tokens": args.tokens, "seconds": elapsed, "tokens_per_second": args.tokens / elapsed}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
