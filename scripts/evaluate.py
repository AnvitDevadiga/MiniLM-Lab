"""Evaluate a saved checkpoint with token-level and tokenizer-independent metrics."""

import argparse
import json
import math
from pathlib import Path

import torch

from minilm_lab.data import load_text, split_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_checkpoint.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer_name = checkpoint.get("tokenizer", "byte")
    tokenizer = ByteTokenizer() if tokenizer_name == "byte" else BPETokenizer.load(tokenizer_name)
    ids = torch.tensor(tokenizer.encode(load_text(args.text)), dtype=torch.long)
    _, validation_ids = split_tokens(ids)
    config = MiniLMConfig(**checkpoint["config"])
    model = MiniLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, validation_ids.numel() - config.context_length - 1, config.context_length):
            x = validation_ids[start : start + config.context_length].unsqueeze(0).to(device)
            y = validation_ids[start + 1 : start + config.context_length + 1].unsqueeze(0).to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    loss = total_loss / total_tokens
    text_bytes = len(load_text(args.text).encode("utf-8"))
    token_count = len(tokenizer.encode(load_text(args.text)))
    result = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": tokenizer_name,
        "device": device,
        "validation_loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2),
        "bits_per_byte": loss * token_count / text_bytes / math.log(2),
        "validation_tokens": total_tokens,
    }
    Path("artifacts/evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
