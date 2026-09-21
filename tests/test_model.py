import torch

from minilm_lab import MiniLM, MiniLMConfig


def test_forward_shapes_and_loss() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=32, context_length=16, embedding_dim=24, num_layers=2, num_heads=4))
    tokens = torch.randint(0, 32, (2, 10))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 10, 32)
    assert loss is not None and torch.isfinite(loss)


def test_context_limit_is_enforced() -> None:
    model = MiniLM(MiniLMConfig(context_length=4, embedding_dim=16, num_layers=1, num_heads=4))
    try:
        model(torch.zeros((1, 5), dtype=torch.long))
    except ValueError as error:
        assert "context_length" in str(error)
    else:
        raise AssertionError("expected a context-length error")


def test_rope_forward_path() -> None:
    config = MiniLMConfig(vocab_size=32, context_length=8, embedding_dim=24, num_layers=1, num_heads=4, positional_encoding="rope")
    logits, loss = MiniLM(config)(torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)
    assert loss is not None
