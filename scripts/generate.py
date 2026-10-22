import argparse
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
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tiny_checkpoint.pt"))
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
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
    print(generate(model, args.prompt, tokenizer, args.tokens, args.temperature, args.top_k, args.top_p))


if __name__ == "__main__":
    main()
