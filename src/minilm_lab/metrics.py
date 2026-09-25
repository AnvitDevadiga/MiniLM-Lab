"""Exact teacher-forced evaluation of a contiguous held-out token stream."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def evaluate_tokens(model, tokens: Tensor, device: str) -> dict[str, float | int]:
    """Score every next token once, using the available left context."""
    if tokens.numel() < 2:
        raise ValueError("validation must contain at least two tokens")
    was_training = model.training
    model.eval()
    total_nll = 0.0
    context = model.config.context_length
    for end in range(1, tokens.numel(), context):
        stop = min(end + context, tokens.numel())
        # One token of left context is shared across chunks. All targets are unique.
        x = tokens[end - 1 : stop - 1].unsqueeze(0).to(device)
        y = tokens[end:stop].unsqueeze(0).to(device)
        logits, _ = model(x)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
    model.train(was_training)
    count = tokens.numel() - 1
    loss = total_nll / count
    return {"validation_loss": loss, "perplexity": math.exp(loss), "bits_per_token": loss / math.log(2), "validation_tokens": count}
