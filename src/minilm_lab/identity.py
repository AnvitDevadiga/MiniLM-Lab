"""Deterministic model identity, independent of checkpoint file bytes.

A checkpoint file's SHA-256 identifies an *artifact*: it changes whenever
runtime metadata such as elapsed seconds changes. The model identity digest
identifies the *model*: architecture configuration, tokenizer and every
tensor's name, dtype, shape and raw values.

Scheme ``minilm-model-identity/v1`` hashes this byte stream with SHA-256:

1. ``b"minilm-model-identity/v1\\n"``
2. canonical JSON (sorted keys, ``separators=(",", ":")``, ASCII, no NaN) of
   ``{"config": <every MiniLMConfig field>, "tokenizer": <identifier>}`` + ``b"\\n"``
3. for each state-dict entry in ascending name order:
   canonical JSON ``{"dtype": "torch.float32", "name": ..., "nbytes": n, "shape": [...]}``
   + ``b"\\n"`` + the tensor's elements in C (row-major) order, each element's
   bytes in little-endian order, in the tensor's own dtype (no casting).

Excluded: elapsed time, metrics, optimizer and RNG state, file paths, timestamps,
seeds and the attention backend (the same weights run on every backend).
"""

import hashlib
import json
import sys
from dataclasses import fields

import torch
from torch import Tensor

from .model import MiniLMConfig

SCHEME = "minilm-model-identity/v1"


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def identity_config(config: dict | MiniLMConfig) -> dict:
    """Every architecture field; legacy configs missing a field take its default."""
    values = config if isinstance(config, MiniLMConfig) else MiniLMConfig(**config)
    return {field.name: getattr(values, field.name) for field in fields(MiniLMConfig)}


def tensor_bytes(tensor: Tensor) -> bytes:
    """Raw element bytes, little-endian, row-major, without changing dtype or the tensor."""
    view = tensor.detach()
    if view.device.type != "cpu":
        view = view.to("cpu")  # a copy; the original tensor is untouched
    flat = view.contiguous().reshape(-1)
    raw = flat.view(torch.uint8) if flat.numel() else torch.empty(0, dtype=torch.uint8)
    if sys.byteorder == "big" and flat.element_size() > 1:
        raw = raw.reshape(-1, flat.element_size()).flip(-1).reshape(-1)
    return raw.numpy().tobytes()


def model_identity(state_dict: dict[str, Tensor], config: dict | MiniLMConfig, tokenizer: str = "byte") -> str:
    digest = hashlib.sha256()
    digest.update(SCHEME.encode("ascii") + b"\n")
    digest.update(_canonical_json({"config": identity_config(config), "tokenizer": tokenizer}) + b"\n")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        header = {"dtype": str(tensor.dtype), "name": name, "nbytes": tensor.numel() * tensor.element_size(), "shape": list(tensor.shape)}
        digest.update(_canonical_json(header) + b"\n")
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def checkpoint_identity(checkpoint: dict) -> dict:
    """Identity of a loaded checkpoint dictionary (current or legacy format)."""
    tokenizer = checkpoint.get("tokenizer", "byte")
    if tokenizer != "byte":
        raise ValueError("model identity is defined for byte-tokenizer checkpoints only")
    return {"scheme": SCHEME, "sha256": model_identity(checkpoint["model"], checkpoint["config"], tokenizer)}
