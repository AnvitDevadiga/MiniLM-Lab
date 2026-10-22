"""Release checks for attention backend defaults, compatibility and tolerances."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import (
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    canonical_backend,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"vocab_size": 64, "context_length": 32, "embedding_dim": 48, "num_layers": 2, "num_heads": 4}
# Maximum absolute logit error against the float64 reference-backend oracle.
# float32 kernels differ in reduction order; float16 is limited by its 10-bit mantissa.
TOLERANCES = {
    ("cpu", torch.float64, "sdpa"): 1e-10,
    ("cpu", torch.float64, "mha"): 1e-10,
    ("cpu", torch.float64, "reference"): 1e-10,
    ("cpu", torch.float32, "sdpa"): 1e-4,
    ("cpu", torch.float32, "mha"): 1e-4,
    ("cpu", torch.float32, "reference"): 1e-4,
    ("mps", torch.float32, "sdpa"): 1e-3,
    ("mps", torch.float32, "mha"): 1e-3,
    ("mps", torch.float32, "reference"): 1e-3,
    ("mps", torch.float16, "sdpa"): 5e-2,
    ("mps", torch.float16, "mha"): 5e-2,
    ("mps", torch.float16, "reference"): 5e-2,
}


def model(positional_encoding: str = "learned", **overrides) -> MiniLM:
    torch.manual_seed(21)
    return MiniLM(MiniLMConfig(**{**CONFIG, **overrides}, positional_encoding=positional_encoding)).eval()


def backends_in_use(m: MiniLM) -> set[str]:
    return {block.attention.backend for block in m.blocks}


def test_sdpa_is_the_default_backend() -> None:
    assert DEFAULT_ATTENTION_BACKEND == "sdpa"
    assert backends_in_use(model()) == {"sdpa"}


@pytest.mark.parametrize(("name", "canonical"), [("reference", "reference"), ("mha", "mha"), ("sdpa", "sdpa"), ("auto", "mha")])
def test_explicit_backend_selection_and_legacy_alias(name: str, canonical: str) -> None:
    assert backends_in_use(model().set_attention_backend(name)) == {canonical}


def test_unknown_or_missing_backend_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        model().set_attention_backend("flash")
    with pytest.raises(ValueError):
        model().set_attention_backend(None)
    # A checkpoint without the field predates it and was trained with mha.
    assert canonical_backend(None) == "mha"


def test_state_dict_is_identical_across_backends() -> None:
    reference = model().state_dict()
    for backend in ATTENTION_BACKENDS:
        state = model().set_attention_backend(backend).state_dict()
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[key], reference[key]) for key in state)


@pytest.mark.parametrize("positional_encoding", ["learned", "rope"])
def test_backend_specific_tolerances(positional_encoding: str) -> None:
    tokens = torch.randint(0, 64, (2, 32), generator=torch.Generator().manual_seed(4))
    base = model(positional_encoding)
    oracle = base.double().set_attention_backend("reference")(tokens)[0]
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for (device, dtype, backend), tolerance in TOLERANCES.items():
        if device not in devices:
            continue
        candidate = model(positional_encoding).to(device=device, dtype=dtype).set_attention_backend(backend)
        error = (candidate(tokens.to(device))[0].cpu().double() - oracle).abs().max().item()
        assert error < tolerance, (device, dtype, backend, error)


@pytest.mark.parametrize("backend", ATTENTION_BACKENDS)
def test_causal_masking_per_backend_with_batch_and_rope(backend: str) -> None:
    m = model("rope").set_attention_backend(backend)
    tokens = torch.randint(0, 64, (3, 32), generator=torch.Generator().manual_seed(5))
    changed = tokens.clone()
    changed[:, 20:] = (changed[:, 20:] + 1) % 64
    assert torch.allclose(m(tokens)[0][:, :20], m(changed)[0][:, :20], atol=1e-6)


def test_evaluation_matches_manual_loss_for_every_backend() -> None:
    ids = torch.randint(0, 64, (200,), generator=torch.Generator().manual_seed(6))
    results = {}
    for backend in ATTENTION_BACKENDS:
        m = model().set_attention_backend(backend)
        total, count = 0.0, 0
        with torch.no_grad():
            for end in range(1, ids.numel(), 32):
                stop = min(end + 32, ids.numel())
                logits, _ = m(ids[end - 1 : stop - 1].unsqueeze(0))
                total += F.cross_entropy(logits[0], ids[end:stop], reduction="sum").item()
                count += stop - end
        evaluation = evaluate_tokens(m, ids, "cpu")
        assert evaluation["validation_tokens"] == count == ids.numel() - 1
        assert abs(evaluation["validation_loss"] - total / count) < 1e-9
        results[backend] = evaluation["validation_loss"]
    assert max(results.values()) - min(results.values()) < 1e-5


@pytest.mark.parametrize("backend", ATTENTION_BACKENDS)
def test_sdpa_dropout_is_disabled_in_evaluation(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    real = F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dropout_p", 0.0))
        return real(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    m = model("rope", dropout=0.3).set_attention_backend(backend)
    tokens = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(7))
    m.eval()
    first, second = m(tokens)[0], m(tokens)[0]
    m.forward_cached(tokens)
    assert torch.equal(first, second)
    if backend != "reference":  # sdpa, and mha with RoPE, call SDPA
        assert seen
    assert all(p == 0.0 for p in seen)
    seen.clear()
    m.train()
    m(tokens)
    if backend != "reference":  # the reference path applies dropout itself
        assert seen and all(p == 0.3 for p in seen)


def train(tmp_path: Path, name: str, extra: list[str]) -> subprocess.CompletedProcess:
    text = tmp_path / "corpus.txt"
    if not text.exists():
        text.write_text("Light is propagated in vacuo with a constant velocity. " * 40, encoding="utf-8")
    command = [
        sys.executable, str(ROOT / "scripts/train_tiny.py"), str(text), "--steps", "2", "--context-length", "16",
        "--embedding-dim", "16", "--num-layers", "1", "--num-heads", "2", "--batch-size", "2", "--eval-interval", "2",
        "--device", "cpu", "--output-dir", str(tmp_path / name), *extra,
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True, cwd=ROOT)


def test_training_defaults_to_sdpa_and_stores_canonical_names(tmp_path: Path) -> None:
    assert train(tmp_path, "default", []).returncode == 0
    assert torch.load(tmp_path / "default/tiny_checkpoint.pt", weights_only=True)["attention_backend"] == "sdpa"
    assert train(tmp_path, "legacy", ["--attention-backend", "auto"]).returncode == 0
    assert torch.load(tmp_path / "legacy/tiny_checkpoint.pt", weights_only=True)["attention_backend"] == "mha"


def test_checkpoints_without_backend_field_resume_only_as_mha(tmp_path: Path) -> None:
    assert train(tmp_path, "old", ["--attention-backend", "mha"]).returncode == 0
    path = tmp_path / "old/tiny_checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["attention_backend"]  # as written before the field existed
    torch.save(checkpoint, path)
    assert train(tmp_path, "old", ["--resume"]).returncode != 0  # sdpa default refused
    resumed = train(tmp_path, "old", ["--resume", "--attention-backend", "mha"])
    assert resumed.returncode == 0, resumed.stderr
    assert torch.load(path, weights_only=True)["step"] == 4


def test_evaluate_script_default_and_explicit_backend(tmp_path: Path) -> None:
    assert train(tmp_path, "run", ["--attention-backend", "mha"]).returncode == 0
    outputs = {}
    for backend in (None, "mha", "reference"):
        out = tmp_path / f"eval-{backend}.json"
        command = [
            sys.executable, str(ROOT / "scripts/evaluate.py"), str(tmp_path / "corpus.txt"), "--device", "cpu",
            "--checkpoint", str(tmp_path / "run/best_checkpoint.pt"), "--output", str(out),
        ] + (["--attention-backend", backend] if backend else [])
        subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
        outputs[backend] = json.loads(out.read_text())
    assert outputs[None]["attention_backend"] == "sdpa"
    assert outputs[None]["checkpoint_attention_backend"] == "mha"
    assert outputs["reference"]["attention_backend"] == "reference"
    losses = [o["validation_loss"] for o in outputs.values()]
    assert max(losses) - min(losses) < 1e-5
