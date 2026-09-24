import torch

from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate_with_kv_cache(model, prompt: str, tokenizer: ByteTokenizer, max_new_tokens: int = 80, temperature: float = 0.8) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, cache = model.forward_cached(tokens[:, -model.config.context_length :])
    for _ in range(max_new_tokens):
        probabilities = torch.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if tokens.size(1) > model.config.context_length:
            # Rebuild the cache from the active sliding window. This keeps
            # long generation requests valid while bounding memory.
            logits, cache = model.forward_cached(tokens[:, -model.config.context_length :])
        else:
            logits, cache = model.forward_cached(next_token, cache)
    return tokenizer.decode(tokens[0].tolist())
