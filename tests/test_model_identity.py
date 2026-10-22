"""Finding 1: deterministic model identity is separate from checkpoint file bytes."""

import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.identity import (
    SCHEME,
    checkpoint_identity,
    identity_config,
    model_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"vocab_size": 32, "context_length": 8, "embedding_dim": 16, "num_layers": 1, "num_heads": 2}


def state(seed: int = 0, **overrides) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    config = MiniLMConfig(**{**CONFIG, **overrides})
    return MiniLM(config).state_dict(), config.__dict__


def test_known_answer_for_the_canonical_byte_stream() -> None:
    tensors = {"b": torch.tensor([1.5, -2.0], dtype=torch.float32), "a": torch.tensor([[1, 2]], dtype=torch.int16)}
    config = identity_config(CONFIG)
    expected = hashlib.sha256()
    expected.update(b"minilm-model-identity/v1\n")
    expected.update(json.dumps({"config": config, "tokenizer": "byte"}, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    expected.update(b'{"dtype":"torch.int16","name":"a","nbytes":4,"shape":[1,2]}\n' + struct.pack("<2h", 1, 2))
    expected.update(b'{"dtype":"torch.float32","name":"b","nbytes":8,"shape":[2]}\n' + struct.pack("<2f", 1.5, -2.0))
    assert model_identity(tensors, CONFIG) == expected.hexdigest()


def test_identical_model_and_config_give_identical_digest() -> None:
    first, config = state(0)
    second, _ = state(0)
    assert model_identity(first, config) == model_identity(second, config)
    assert model_identity(first, config) == model_identity(dict(reversed(list(first.items()))), config)


def test_weight_change_changes_digest() -> None:
    weights, config = state(0)
    changed = {name: tensor.clone() for name, tensor in weights.items()}
    changed["blocks.0.up.weight"][0, 0] += 1e-6
    assert model_identity(weights, config) != model_identity(changed, config)
    assert model_identity(weights, config) != model_identity(state(1)[0], config)


@pytest.mark.parametrize("override", [{"dropout": 0.1}, {"rope_theta": 500.0}, {"positional_encoding": "rope"}, {"norm": "layernorm"}])
def test_identity_relevant_config_changes_digest(override: dict) -> None:
    weights, config = state(0)
    assert model_identity(weights, config) != model_identity(weights, {**config, **override})


def test_tokenizer_is_part_of_identity() -> None:
    weights, config = state(0)
    assert model_identity(weights, config, "byte") != model_identity(weights, config, "bpe-v1")


def test_runtime_metadata_does_not_change_digest() -> None:
    weights, config = state(0)
    base = {"model": weights, "config": config, "tokenizer": "byte", "seed": 1,
            "metrics": [{"step": 1, "validation_loss": 2.0, "elapsed_seconds": 1.0, "train_seconds": 0.5, "eval_seconds": 0.1}]}
    later = copy.deepcopy(base)
    later["metrics"][0].update(elapsed_seconds=99.0, train_seconds=42.0, eval_seconds=7.0)
    later.update(seed=2, attention_backend="sdpa", output_dir="/elsewhere", created="2026-10-22T05:00:00Z")
    assert checkpoint_identity(base) == checkpoint_identity(later)
    assert checkpoint_identity(base)["scheme"] == SCHEME


def test_precision_is_preserved_not_cast() -> None:
    weights, config = state(0)
    half = {name: tensor.to(torch.float16) if tensor.is_floating_point() else tensor for name, tensor in weights.items()}
    bf16 = {name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor for name, tensor in weights.items()}
    digests = {model_identity(weights, config), model_identity(half, config), model_identity(bf16, config)}
    assert len(digests) == 3


def test_hashing_does_not_mutate_model_state() -> None:
    torch.manual_seed(0)
    model = MiniLM(MiniLMConfig(**CONFIG)).train()
    before = {name: (tensor.clone(), tensor._version, tensor.requires_grad, tensor.dtype, tensor.device) for name, tensor in model.state_dict(keep_vars=True).items()}
    model_identity(model.state_dict(keep_vars=True), model.config)
    after = model.state_dict(keep_vars=True)
    for name, (value, version, requires_grad, dtype, device) in before.items():
        assert torch.equal(after[name], value) and after[name]._version == version
        assert (after[name].requires_grad, after[name].dtype, after[name].device) == (requires_grad, dtype, device)
    assert model.training


def test_device_does_not_change_digest_or_move_tensors(mps_device: str) -> None:
    weights, config = state(0)
    on_mps = {name: tensor.to("mps") for name, tensor in weights.items()}
    assert model_identity(on_mps, config) == model_identity(weights, config)
    assert all(tensor.device.type == "mps" for tensor in on_mps.values())


def test_legacy_checkpoint_without_identity_or_new_config_fields_loads_and_hashes(tmp_path: Path) -> None:
    weights, config = state(0)
    legacy_config = {key: value for key, value in config.items() if key not in ("rope_theta", "norm", "activation")}
    path = tmp_path / "legacy.pt"
    torch.save({"model": weights, "config": legacy_config, "metrics": [], "seed": 42}, path)
    loaded = torch.load(path, weights_only=True)
    model = MiniLM(MiniLMConfig(**loaded["config"]))
    model.load_state_dict(loaded["model"])
    assert checkpoint_identity(loaded)["sha256"] == model_identity(weights, config)


def test_training_stores_identity_that_matches_tensors_across_reruns(tmp_path: Path) -> None:
    text = tmp_path / "corpus.txt"
    text.write_text("Every candle burns with a flame of its own shape. " * 30, encoding="utf-8")
    identities = []
    for run in ("a", "b"):
        command = [sys.executable, str(ROOT / "scripts/train_tiny.py"), str(text), "--steps", "3", "--context-length", "16",
                   "--embedding-dim", "16", "--num-layers", "1", "--num-heads", "2", "--batch-size", "2", "--eval-interval", "1",
                   "--device", "cpu", "--seed", "5", "--output-dir", str(tmp_path / run)]
        subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
        for name in ("best_checkpoint.pt", "tiny_checkpoint.pt"):
            checkpoint = torch.load(tmp_path / run / name, weights_only=True)
            assert checkpoint["model_identity"] == checkpoint_identity(checkpoint)
        identities.append(torch.load(tmp_path / run / "tiny_checkpoint.pt", weights_only=True)["model_identity"])
    # Same seed and data: the model identity is reproduced. File bytes are not compared:
    # they include elapsed seconds and are not expected to match.
    assert identities[0] == identities[1]
