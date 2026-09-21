import torch

from minilm_lab import MiniLM, MiniLMConfig


def test_kv_cache_shapes() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=32, context_length=16, embedding_dim=24, num_layers=2, num_heads=4))
    logits, cache = model.forward_cached(torch.randint(0, 32, (1, 5)))
    next_logits, next_cache = model.forward_cached(torch.randint(0, 32, (1, 1)), cache)
    assert logits.shape == (1, 5, 32)
    assert next_logits.shape == (1, 1, 32)
    assert next_cache[0][0].shape[2] == 6
