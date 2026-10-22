import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.kv_cache import greedy_decode

ROOT = Path(__file__).resolve().parents[1]


def model_for(positional_encoding: str, context_length: int = 16) -> MiniLM:
    torch.manual_seed(5)
    config = MiniLMConfig(
        vocab_size=48,
        context_length=context_length,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        positional_encoding=positional_encoding,
    )
    return MiniLM(config).eval()


@pytest.mark.parametrize("positional_encoding", ["learned", "rope"])
@pytest.mark.parametrize(
    ("batch", "prompt_length", "new_tokens"),
    [
        (1, 1, 1),  # single-token prompt, single generated token
        (1, 1, 15),  # fill the context exactly
        (4, 5, 8),  # batched decode
        (2, 16, 6),  # prompt already at the context limit
        (3, 3, 40),  # generation well past the window (sliding refresh)
    ],
)
def test_cached_greedy_decode_matches_uncached(positional_encoding: str, batch: int, prompt_length: int, new_tokens: int) -> None:
    model = model_for(positional_encoding)
    prompt = torch.randint(0, 48, (batch, prompt_length))
    uncached = greedy_decode(model, prompt, new_tokens, use_cache=False)
    cached = greedy_decode(model, prompt, new_tokens, use_cache=True)
    assert cached.shape == (batch, prompt_length + new_tokens)
    assert torch.equal(cached, uncached)


def test_cached_batch_rows_match_individual_rows() -> None:
    model = model_for("rope")
    prompt = torch.randint(0, 48, (3, 4))
    batched = greedy_decode(model, prompt, 10)
    for row in range(3):
        assert torch.equal(batched[row : row + 1], greedy_decode(model, prompt[row : row + 1], 10))


def test_cache_grows_by_one_position_per_decoded_token() -> None:
    model = model_for("learned")
    logits, cache = model.forward_cached(torch.randint(0, 48, (2, 3)))
    sizes = [cache[0][0].size(2)]
    for _ in range(5):
        logits, cache = model.forward_cached(logits[:, -1:].argmax(-1), cache)
        sizes.append(cache[0][0].size(2))
    assert sizes == [3, 4, 5, 6, 7, 8]
    # keys and values, every layer, (batch, heads, positions, head_width)
    assert all(k.shape == v.shape == (2, 4, 8, 8) for k, v in cache)


def test_gradient_accumulation_matches_full_batch() -> None:
    model = model_for("learned").train()
    inputs = torch.randint(0, 48, (8, 16))
    targets = torch.randint(0, 48, (8, 16))
    _, loss = model(inputs, targets)
    loss.backward()
    full = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    # Same arithmetic as scripts/train_tiny.py: each microbatch loss / steps.
    for chunk in range(2):
        _, loss = model(inputs[chunk * 4 : (chunk + 1) * 4], targets[chunk * 4 : (chunk + 1) * 4])
        (loss / 2).backward()
    for name, parameter in model.named_parameters():
        if name in full:
            assert torch.allclose(parameter.grad, full[name], atol=1e-6, rtol=1e-4), name


def run_training(text: Path, output: Path, steps: int, resume: bool = False) -> dict:
    command = [
        sys.executable, str(ROOT / "scripts/train_tiny.py"), str(text), "--steps", str(steps),
        "--context-length", "16", "--embedding-dim", "16", "--num-layers", "1", "--num-heads", "2",
        "--batch-size", "2", "--eval-interval", "2", "--gradient-accumulation-steps", "2",
        "--device", "cpu", "--seed", "3", "--output-dir", str(output),
    ]
    subprocess.run(command + (["--resume"] if resume else []), check=True, capture_output=True, cwd=ROOT)
    return torch.load(output / "tiny_checkpoint.pt", weights_only=True)


def test_resume_reproduces_uninterrupted_training(tmp_path: Path) -> None:
    text = tmp_path / "corpus.txt"
    text.write_text("The quick brown fox jumps over the lazy dog. " * 40, encoding="utf-8")
    straight = run_training(text, tmp_path / "straight", steps=6)
    run_training(text, tmp_path / "resumed", steps=3)
    resumed = run_training(text, tmp_path / "resumed", steps=3, resume=True)
    assert resumed["step"] == straight["step"] == 6
    for name, tensor in straight["model"].items():
        assert torch.equal(tensor, resumed["model"][name]), name
    # A segment always ends with an evaluation, so the resumed history has an
    # extra step-3 record. Shared steps must agree exactly.
    resumed_by_step = {m["step"]: m["validation_loss"] for m in resumed["metrics"]}
    assert sorted(resumed_by_step) == [2, 3, 4, 6]
    assert all(resumed_by_step[m["step"]] == m["validation_loss"] for m in straight["metrics"])
