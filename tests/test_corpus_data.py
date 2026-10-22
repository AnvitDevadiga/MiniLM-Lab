"""Integrity and contamination checks on the committed corpora."""

import hashlib
import json
from pathlib import Path

import pytest

from minilm_lab.corpus import contamination, paragraphs

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/science"
V2 = ROOT / "data/science-v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_v1_corpus_is_unchanged() -> None:
    manifest = json.loads((V1 / "manifest.json").read_text())
    assert sha256(V1 / "train.txt") == manifest["train_sha256"]
    assert sha256(V1 / "validation.txt") == manifest["validation_sha256"]


@pytest.fixture(scope="module")
def v2() -> dict:
    return json.loads((V2 / "manifest.json").read_text())


def test_v2_files_match_manifest(v2: dict) -> None:
    for name, info in v2["files"].items():
        assert sha256(V2 / name) == info["sha256"], name
        assert (V2 / name).stat().st_size == info["bytes"], name


def test_v2_test_book_is_a_separate_author(v2: dict) -> None:
    roles = {s["role"]: s for s in v2["sources"] if s["role"] == "test"}
    train_authors = {s["author"] for s in v2["sources"] if s["role"] == "train_validation"}
    assert roles["test"]["author"] not in train_authors


def test_v2_held_out_text_has_no_exact_or_near_duplicates() -> None:
    train = paragraphs((V2 / "train.txt").read_text())
    validation = paragraphs((V2 / "validation.txt").read_text())
    test = paragraphs((V2 / "test.txt").read_text())
    for reference, candidates in ((train, validation), (train, test), (validation, test)):
        report = contamination(reference, candidates)
        assert report["exact_duplicates"] == [] and report["near_duplicates"] == []


def test_v2_splits_contain_no_index_or_backmatter() -> None:
    for name in ("train.txt", "validation.txt", "test.txt"):
        text = (V2 / name).read_text()
        assert "\nINDEX.\n" not in text and "Aberrant groups, 429." not in text
        assert "[Illustration" not in text and "_K_ = co-ordinate system" not in text
        assert "PROJECT GUTENBERG" not in text.upper()


def test_v2_per_book_validation_files_concatenate_to_validation() -> None:
    parts = [(V2 / f"{s['name']}-validation.txt").read_text() for s in json.loads((V2 / "manifest.json").read_text())["sources"] if s["role"] == "train_validation"]
    # Same joining as v1 and prepare_science_corpus_v2.py.
    assert "\n\n".join(parts).strip() + "\n" == (V2 / "validation.txt").read_text()
