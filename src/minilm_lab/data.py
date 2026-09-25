from pathlib import Path

import torch
from torch import Tensor


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_tokens(tokens: Tensor, validation_fraction: float = 0.1) -> tuple[Tensor, Tensor]:
    split = int(tokens.numel() * (1.0 - validation_fraction))
    if split <= 1 or tokens.numel() - split <= 1:
        raise ValueError("text must contain enough tokens for a train/validation split")
    return tokens[:split], tokens[split:]


def make_batch(tokens: Tensor, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    if tokens.numel() <= context_length:
        raise ValueError("split must contain more tokens than context_length")
    starts = torch.randint(0, tokens.numel() - context_length, (batch_size,))
    x = torch.stack([tokens[i : i + context_length] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)
