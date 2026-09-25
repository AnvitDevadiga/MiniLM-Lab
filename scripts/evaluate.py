"""Evaluate a saved checkpoint with token-level and tokenizer-independent metrics."""

import argparse
import json
import math
from pathlib import Path

import torch

from minilm_lab.data import load_text, split_tokens
from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--validation-text", type=Path, help="evaluate a pre-split held-out file")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_checkpoint.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer_name = checkpoint.get("tokenizer", "byte")
    tokenizer = ByteTokenizer() if tokenizer_name == "byte" else BPETokenizer.load(tokenizer_name)
    if args.validation_text:
        validation_ids = torch.tensor(tokenizer.encode(load_text(args.validation_text)), dtype=torch.long)
    else:
        ids = torch.tensor(tokenizer.encode(load_text(args.text)), dtype=torch.long)
        _, validation_ids = split_tokens(ids)
    config = MiniLMConfig(**checkpoint["config"])
    model = MiniLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    evaluation = evaluate_tokens(model, validation_ids, device)
    # The first token is context; count bytes represented by scored targets.
    scored_bytes = len(tokenizer.decode_bytes(validation_ids[1:].tolist()))
    result = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": tokenizer_name,
        "device": device,
        **evaluation,
        "bits_per_byte": evaluation["validation_loss"] * evaluation["validation_tokens"] / scored_bytes / math.log(2),
        "validation_bytes": scored_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
