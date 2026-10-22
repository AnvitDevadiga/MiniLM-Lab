import pytest
import torch
import torch.nn.functional as F

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.model import (
    CausalSelfAttention,
    causal_mask,
    reference_causal_attention,
)

BACKENDS = ("auto", "sdpa", "reference")


def small_model(positional_encoding: str = "learned", context_length: int = 32) -> MiniLM:
    torch.manual_seed(11)
    config = MiniLMConfig(
        vocab_size=64,
        context_length=context_length,
        embedding_dim=48,
        num_layers=2,
        num_heads=4,
        positional_encoding=positional_encoding,
    )
    return MiniLM(config).eval()


def test_causal_mask_is_bottom_right_aligned() -> None:
    assert torch.equal(causal_mask(3, 3, torch.device("cpu")), torch.ones(3, 3).tril().bool())
    # Two cached keys, two new queries: query 0 is absolute position 2.
    expected = torch.tensor([[True, True, True, False], [True, True, True, True]])
    assert torch.equal(causal_mask(2, 4, torch.device("cpu")), expected)


def test_reference_attention_keeps_float64_precision() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 8, dtype=torch.float64) for _ in range(3))
    output = reference_causal_attention(q, k, v)
    assert output.dtype == torch.float64
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (output - expected).abs().max() < 1e-12


def test_reference_attention_matches_hand_computed_softmax() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 8, dtype=torch.float64) for _ in range(3))
    output = reference_causal_attention(q, k, v)
    for row in range(5):
        weights = torch.softmax(q[:, :, row : row + 1] @ k[:, :, : row + 1].transpose(-2, -1) / 8**0.5, dim=-1)
        assert torch.allclose(output[:, :, row : row + 1], weights @ v[:, :, : row + 1])


@pytest.mark.parametrize("positional_encoding", ["learned", "rope"])
def test_backends_match_float64_oracle(positional_encoding: str) -> None:
    model = small_model(positional_encoding)
    tokens = torch.randint(0, 64, (3, 32))
    oracle = model.double().set_attention_backend("reference")(tokens)[0]
    model.float()
    for backend in BACKENDS:
        logits, _ = model.set_attention_backend(backend)(tokens)
        assert logits.shape == (3, 32, 64)
        assert (logits.double() - oracle).abs().max() < 1e-4, backend


@pytest.mark.parametrize("positional_encoding", ["learned", "rope"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_future_tokens_do_not_change_past_logits(positional_encoding: str, backend: str) -> None:
    model = small_model(positional_encoding).set_attention_backend(backend)
    tokens = torch.randint(0, 64, (2, 20))
    changed = tokens.clone()
    changed[:, 12:] = torch.randint(0, 64, (2, 8))
    original, _ = model(tokens)
    perturbed, _ = model(changed)
    assert torch.allclose(original[:, :12], perturbed[:, :12], atol=1e-6)
    assert not torch.allclose(original[:, 12:], perturbed[:, 12:])


@pytest.mark.parametrize("backend", ["sdpa", "reference"])
def test_cached_backends_match_full_forward_in_chunks(backend: str) -> None:
    model = small_model("rope").set_attention_backend(backend)
    tokens = torch.randint(0, 64, (3, 32))
    full, _ = model(tokens)
    pieces, cache = [], None
    for start, stop in ((0, 1), (1, 9), (9, 10), (10, 32)):
        logits, cache = model.forward_cached(tokens[:, start:stop], cache)
        pieces.append(logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)
    assert cache[0][0].shape == (3, 4, 32, 12)


def test_backend_selection_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        small_model().set_attention_backend("flash")


def test_backend_does_not_change_weights() -> None:
    model = small_model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.set_attention_backend("reference")
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def rope_layer(width: int = 16) -> CausalSelfAttention:
    return CausalSelfAttention(MiniLMConfig(context_length=64, embedding_dim=width * 2, num_heads=2))


def test_rope_matches_complex_rotation() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 1, 10, 16, dtype=torch.float64)
    rotated = rope_layer()._apply_rope(x, position_offset=3)
    frequencies = 1.0 / (10000.0 ** (torch.arange(8, dtype=torch.float64) / 8))
    angles = torch.arange(3, 13, dtype=torch.float64)[:, None] * frequencies[None, :]
    expected = torch.complex(x[..., :8], x[..., 8:]) * torch.polar(torch.ones_like(angles), angles)
    assert torch.allclose(rotated, torch.cat((expected.real, expected.imag), dim=-1))


def test_rope_scores_depend_only_on_relative_position() -> None:
    torch.manual_seed(2)
    layer = rope_layer()
    q = torch.randn(1, 1, 1, 16, dtype=torch.float64).expand(1, 1, 40, 16)
    k = torch.randn(1, 1, 1, 16, dtype=torch.float64).expand(1, 1, 40, 16)
    scores = layer._apply_rope(q) @ layer._apply_rope(k).transpose(-2, -1)
    # Same offset m - n gives the same score anywhere in the sequence.
    assert torch.allclose(scores[0, 0, 7, 3], scores[0, 0, 30, 26])
    assert torch.allclose(layer._apply_rope(q).norm(dim=-1), q.norm(dim=-1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_half_precision_error_is_at_rounding_level(dtype: torch.dtype) -> None:
    torch.manual_seed(3)
    layer = CausalSelfAttention(MiniLMConfig(context_length=2048, embedding_dim=192, num_heads=6))
    x = torch.randn(1, 6, 2048, 32)
    exact = layer._apply_rope(x.double())
    error = (layer._apply_rope(x.to(dtype)).double() - exact).abs().max()
    rounding = (exact.to(dtype).double() - exact).abs().max()
    # Before computing angles in float32, bf16 error at 2048 positions was ~7.3.
    assert error < 4 * rounding


@pytest.mark.parametrize("positional_encoding", ["learned", "rope"])
def test_backend_gradients_match_float64_oracle(positional_encoding: str) -> None:
    model = small_model(positional_encoding).train()
    tokens = torch.randint(0, 64, (3, 32))
    targets = torch.randint(0, 64, (3, 32))

    def gradients(backend: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        model.to(dtype).set_attention_backend(backend).zero_grad(set_to_none=True)
        model(tokens, targets)[1].backward()
        return {name: p.grad.double().clone() for name, p in model.named_parameters() if p.grad is not None}

    oracle = gradients("reference", torch.float64)
    for backend in BACKENDS:
        grads = gradients(backend, torch.float32)
        assert grads.keys() == oracle.keys()
        for name, grad in grads.items():
            scale = oracle[name].abs().max().clamp_min(1e-8)
            assert (grad - oracle[name]).abs().max() / scale < 1e-4, (backend, name)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_rope", [False, True])
def test_attention_layer_passes_gradcheck(backend: str, use_rope: bool) -> None:
    if backend == "auto" and use_rope:
        pytest.skip("auto uses the SDPA path for RoPE, covered by sdpa")
    torch.manual_seed(6)
    layer = CausalSelfAttention(MiniLMConfig(context_length=8, embedding_dim=8, num_heads=2)).double()
    layer.backend = backend
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: layer(value, use_rope=use_rope), (x,))


@pytest.mark.parametrize("backend", BACKENDS)
def test_attention_dropout_only_in_training(backend: str) -> None:
    torch.manual_seed(8)
    config = MiniLMConfig(vocab_size=64, context_length=16, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.5)
    model = MiniLM(config).set_attention_backend(backend)
    tokens = torch.randint(0, 64, (2, 16))
    model.eval()
    assert torch.equal(model(tokens)[0], model(tokens)[0])
    model.train()
    attention = model.blocks[0].attention
    x = torch.randn(2, 16, 32)
    assert not torch.allclose(attention(x), attention(x))
