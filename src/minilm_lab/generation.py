import torch

from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate(model, prompt: str, tokenizer: ByteTokenizer, max_new_tokens: int = 80, temperature: float = 0.8, top_k: int | None = 40, top_p: float = 1.0) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.context_length :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokenizer.decode(tokens[0].tolist())
