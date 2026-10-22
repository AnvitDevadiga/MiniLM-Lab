import torch

from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate_with_kv_cache(model, prompt: str, tokenizer: ByteTokenizer, max_new_tokens: int = 80, temperature: float = 0.8) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, cache = model.forward_cached(tokens[:, -model.config.context_length :])
    for index in range(max_new_tokens):
        probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if index == max_new_tokens - 1:
            break
        if tokens.size(1) > model.config.context_length:
            # Rebuild the cache from the active sliding window. This keeps
            # long generation requests valid while bounding memory.
            logits, cache = model.forward_cached(tokens[:, -model.config.context_length :])
        else:
            logits, cache = model.forward_cached(next_token, cache)
    return tokenizer.decode(tokens[0].tolist())


@torch.no_grad()
def greedy_decode(model, input_ids: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
    """Deterministic batched argmax decoding, with or without the KV cache.

    Both modes see the same sliding window of the last context_length tokens,
    so their outputs are directly comparable.
    """
    if input_ids.dim() != 2 or input_ids.size(1) < 1:
        raise ValueError("input_ids must have shape (batch, length >= 1)")
    model.eval()
    window = model.config.context_length
    tokens = input_ids
    cache = None
    for index in range(max_new_tokens):
        if not use_cache:
            logits, _ = model(tokens[:, -window:])
        elif cache is None or tokens.size(1) > window:
            logits, cache = model.forward_cached(tokens[:, -window:])
        else:
            logits, cache = model.forward_cached(tokens[:, -1:], cache)
        tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return tokens
