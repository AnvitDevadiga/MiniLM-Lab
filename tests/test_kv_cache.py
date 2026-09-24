import torch

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.kv_cache import generate_with_kv_cache
from minilm_lab.tokenizer import ByteTokenizer


def test_kv_cache_shapes() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=32, context_length=16, embedding_dim=24, num_layers=2, num_heads=4))
    logits, cache = model.forward_cached(torch.randint(0, 32, (1, 5)))
    next_logits, next_cache = model.forward_cached(torch.randint(0, 32, (1, 1)), cache)
    assert logits.shape == (1, 5, 32)
    assert next_logits.shape == (1, 1, 32)
    assert next_cache[0][0].shape[2] == 6


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(7)
    for positional_encoding in ("learned", "rope"):
        config = MiniLMConfig(
            vocab_size=32,
            context_length=16,
            embedding_dim=24,
            num_layers=2,
            num_heads=4,
            positional_encoding=positional_encoding,
        )
        model = MiniLM(config).eval()
        prompt = torch.randint(0, 32, (1, 6))
        continuation = torch.randint(0, 32, (1, 1))
        full_logits, _ = model(torch.cat((prompt, continuation), dim=1))
        _, cache = model.forward_cached(prompt)
        cached_logits, _ = model.forward_cached(continuation, cache)
        assert torch.allclose(full_logits[:, -1], cached_logits[:, -1], atol=1e-5, rtol=1e-5)


def test_kv_cache_context_limit_is_enforced() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=16, context_length=4, embedding_dim=16, num_layers=1, num_heads=4))
    _, cache = model.forward_cached(torch.randint(0, 16, (1, 4)))
    try:
        model.forward_cached(torch.randint(0, 16, (1, 1)), cache)
    except ValueError as error:
        assert "context_length" in str(error)
    else:
        raise AssertionError("expected a cached context-length error")


def test_cached_generation_supports_sliding_window() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=256, context_length=8, embedding_dim=16, num_layers=1, num_heads=4)).eval()
    result = generate_with_kv_cache(model, "Hi", ByteTokenizer(), max_new_tokens=12)
    assert result.startswith("Hi")
