"""Create a CPU dynamic-int8 quantized checkpoint for comparison."""

import argparse
from pathlib import Path

import torch
from torch import nn

from minilm_lab.model import MiniLM, MiniLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/quantized_checkpoint.pt"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MiniLM(MiniLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    args.output.parent.mkdir(exist_ok=True)
    torch.save({"model": quantized.state_dict(), "config": checkpoint["config"], "tokenizer": checkpoint.get("tokenizer", "byte")}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
