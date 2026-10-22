"""Regression tests for the three release-blocking defects found in ad71757.

Fix 1: model identities must be bound to checkpoint artifacts or a pinned audited baseline.
Fix 2: duplicate experiments are detected after backend canonicalization and identity resolution.
Fix 3: KV-cache speedups are derived from timings; stored ratios are checked, never trusted.

All tests use copies of experiments/records. Tests that need the original checkpoint
files skip when they are absent (fresh clones); every other case runs everywhere.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minilm_lab import MiniLM, MiniLMConfig
from minilm_lab.identity import checkpoint_identity
from minilm_lab.records import (
    ARTIFACT_VERIFIED,
    BASELINE_PATH,
    BASELINE_VERIFIED,
    METADATA_ONLY,
    UNVERIFIED,
    RecordError,
    identity_evidence,
    kv_speedups,
    load_store,
    protocol_rows,
    resolve_identity,
    validate_store,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"
ATTESTATION = "identity/20261022T053914Z_model_identity_attestation_935a229.json"
V1_42 = "training/20261022T041654Z_training_science-v1_auto_seed42_48e95f8.json"
V1_43 = "training/20261022T041103Z_training_science-v1_auto_seed43_48e95f8.json"
V1_44 = "training/20261022T041335Z_training_science-v1_auto_seed44_48e95f8.json"
KV = "kv_cache/20261022T053124Z_kv_cpu_float32_57be26d.json"
HISTORICAL_KV = "kv_cache/20261022T035538Z_kv_cpu_float32_93507c4.json"
LEGACY_KV = "legacy/20261022T045428Z_legacy_kv_cache_cpu_13530f4.json"
ARTIFACTS_PRESENT = (ROOT / "artifacts/runs/sdpa-v1/auto-s42/best_checkpoint.pt").is_file()


@pytest.fixture
def records(tmp_path: Path) -> Path:
    copy = tmp_path / "records"
    shutil.copytree(RECORDS, copy)
    return copy


def read(root: Path, key: str) -> dict:
    return json.loads((root / key).read_text())


def write(root: Path, key: str, record: dict) -> None:
    (root / key).write_text(json.dumps(record, indent=2))


def edit(root: Path, key: str, change) -> None:
    record = read(root, key)
    change(record)
    write(root, key, record)


def problems(root: Path, **options) -> list[str]:
    strict = options.pop("strict", False)
    return validate_store(load_store(ROOT, root, **options), strict=strict)


def attested_row(record: dict, training_key: str, kind: str = "best") -> dict:
    return next(r for r in record["results"]["attested"] if r["training_record"] == training_key and r["kind"] == kind)


# ================================================================ Fix 1: artifact / attestation integrity


@pytest.mark.parametrize("artifacts", ["present", "absent"])
def test_zeroed_attestation_fingerprint_fails(records: Path, tmp_path: Path, artifacts: str) -> None:
    edit(records, ATTESTATION, lambda r: attested_row(r, V1_43).update(model_identity_sha256="0" * 64))
    options = {} if artifacts == "present" else {"artifact_root": tmp_path / "empty"}
    assert any("model identity sources disagree" in p and V1_43 in p for p in problems(records, **options))


def test_declared_fingerprint_contradicting_attestation_fails(records: Path, tmp_path: Path) -> None:
    edit(records, V1_43, lambda r: r["results"].update(best_model_identity_sha256="a" * 64))
    # Even with no artifact and no baseline, declared vs attested must agree.
    found = problems(records, artifact_root=tmp_path / "empty", baseline_path=None)
    assert any("model identity sources disagree" in p and "declared=aaaaaaaaaaaa" in p for p in found)


def test_attestation_with_another_checkpoints_fingerprint_fails(records: Path) -> None:
    other = attested_row(read(RECORDS, ATTESTATION), V1_44)["model_identity_sha256"]
    edit(records, ATTESTATION, lambda r: attested_row(r, V1_43).update(model_identity_sha256=other))
    assert any("model identity sources disagree" in p and V1_43 in p for p in problems(records))


def test_changed_checkpoint_bytes_without_updated_hash_fail(records: Path, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts-root"
    checkpoint = artifact_root / "artifacts/runs/sdpa-v1/auto-s43/best_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"tampered checkpoint bytes")
    found = problems(records, artifact_root=artifact_root)
    assert any("do not match the recorded SHA-256" in p and V1_43 in p for p in found)
    evidence, _ = resolve_identity(load_store(ROOT, records, artifact_root=artifact_root), V1_43)
    assert evidence.status == UNVERIFIED and evidence.fingerprint is None


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="original checkpoint artifacts not in this checkout")
def test_flipped_byte_in_real_checkpoint_fails(records: Path, tmp_path: Path) -> None:
    source = ROOT / "artifacts/runs/sdpa-v1/auto-s43/best_checkpoint.pt"
    target = tmp_path / "root/artifacts/runs/sdpa-v1/auto-s43/best_checkpoint.pt"
    target.parent.mkdir(parents=True)
    data = bytearray(source.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    assert any("do not match the recorded SHA-256" in p for p in problems(records, artifact_root=tmp_path / "root"))


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="original checkpoint artifacts not in this checkout")
def test_recomputing_the_historical_baseline_from_artifacts_reproduces_it() -> None:
    run = subprocess.run([sys.executable, str(ROOT / "scripts/build_identity_baseline.py"), "--check"], check=False, capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr
    levels = {e.status for e in identity_evidence(load_store(ROOT), []).values()}
    assert levels == {ARTIFACT_VERIFIED}


def test_legitimate_historical_attestation_agrees_with_pinned_baseline(tmp_path: Path) -> None:
    baseline = json.loads((ROOT / BASELINE_PATH).read_text())["entries"]
    for row in read(RECORDS, ATTESTATION)["results"]["attested"]:
        assert baseline[row["checkpoint_sha256"]]["model_identity_sha256"] == row["model_identity_sha256"]
    store = load_store(ROOT, artifact_root=tmp_path / "empty")
    assert validate_store(store, strict=True) == []
    assert {e.status for e in identity_evidence(store, []).values()} == {BASELINE_VERIFIED}


def test_unavailable_artifacts_and_baseline_are_never_reported_as_verified(records: Path, tmp_path: Path) -> None:
    store = load_store(ROOT, records, artifact_root=tmp_path / "empty", baseline_path=None)
    assert validate_store(store) == []  # internally consistent metadata...
    assert {e.status for e in identity_evidence(store, []).values()} == {METADATA_ONLY}  # ...is only that
    strict = validate_store(load_store(ROOT, records, artifact_root=tmp_path / "empty", baseline_path=None), strict=True)
    assert strict and all("metadata_consistent_only" in p for p in strict)
    # A forged attestation cannot be detected without independent evidence, so strict must fail.
    edit(records, ATTESTATION, lambda r: attested_row(r, V1_43).update(model_identity_sha256="0" * 64))
    assert problems(records, artifact_root=tmp_path / "empty", baseline_path=None, strict=True)


def test_validator_cli_reports_evidence_levels_and_fails_closed(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts/validate_records.py"), "--artifact-root", str(tmp_path)]
    baseline = subprocess.run(command + ["--strict"], check=False, capture_output=True, text=True, cwd=ROOT)
    assert baseline.returncode == 0 and f"{BASELINE_VERIFIED}=24" in baseline.stdout and ARTIFACT_VERIFIED not in baseline.stdout
    metadata = subprocess.run(command + ["--strict", "--no-baseline"], check=False, capture_output=True, text=True, cwd=ROOT)
    assert metadata.returncode == 1 and f"{METADATA_ONLY}=24" in metadata.stdout
    artifacts = subprocess.run(command + ["--require-artifacts"], check=False, capture_output=True, text=True, cwd=ROOT)
    assert artifacts.returncode == 1


def test_tampered_baseline_is_not_trusted(tmp_path: Path) -> None:
    baseline = json.loads((ROOT / BASELINE_PATH).read_text())
    first = next(iter(baseline["entries"]))
    baseline["entries"][first]["model_identity_sha256"] = "0" * 64
    forged = tmp_path / "baseline.json"
    forged.write_text(json.dumps(baseline))
    found = validate_store(load_store(ROOT, artifact_root=tmp_path / "empty", baseline_path=forged), strict=True)
    assert any("does not match the pinned audited baseline" in p for p in found)


def synthetic_run(tmp_path: Path, records: Path, declared: str | None) -> tuple[Path, str, str]:
    """A modern record pointing at a real, freshly written checkpoint artifact."""
    torch.manual_seed(0)
    model = MiniLM(MiniLMConfig(vocab_size=32, context_length=8, embedding_dim=16, num_layers=1, num_heads=2))
    checkpoint = {"model": model.state_dict(), "config": model.config.__dict__, "tokenizer": "byte", "metrics": []}
    output = "artifacts/runs/synthetic"
    root = tmp_path / "root"
    (root / output).mkdir(parents=True)
    for name in ("best_checkpoint.pt", "tiny_checkpoint.pt"):
        torch.save(checkpoint, root / output / name)
    import hashlib

    sha = hashlib.sha256((root / output / "best_checkpoint.pt").read_bytes()).hexdigest()
    record = read(records, V1_42)
    record["created_utc"] = "20261022T090000Z"
    record["record_version"] = 2
    record["settings"]["training_command"] = f"python scripts/train_tiny.py data/science/train.txt --output-dir {output}"
    record["results"].update(best_checkpoint_sha256=sha, final_checkpoint_sha256=sha, best_model_identity_sha256=declared,
                             final_model_identity_sha256=declared)
    key = "training/20261022T090000Z_training_science-v1_mha_seed99_48e95f8.json"
    record["settings"]["seed"] = 99
    write(records, key, record)
    return root, key, checkpoint_identity(checkpoint)["sha256"]


def test_correct_modern_record_is_artifact_verified(records: Path, tmp_path: Path) -> None:
    root, key, identity = synthetic_run(tmp_path, records, declared=None)
    edit(records, key, lambda r: r["results"].update(best_model_identity_sha256=identity, final_model_identity_sha256=identity))
    evidence, issues = resolve_identity(load_store(ROOT, records, artifact_root=root), key)
    assert issues == [] and evidence.status == ARTIFACT_VERIFIED and evidence.fingerprint == identity


def test_modern_record_whose_declared_identity_differs_from_artifact_fails(records: Path, tmp_path: Path) -> None:
    root, key, _ = synthetic_run(tmp_path, records, declared="d" * 64)
    evidence, issues = resolve_identity(load_store(ROOT, records, artifact_root=root), key)
    assert evidence.status == UNVERIFIED and any("sources disagree" in p for p in issues)


# ================================================================ Fix 2: canonical duplicate integrity


def add_duplicate(root: Path, base: str = V1_42, stamp: str = "20261022T060000Z", change=None) -> str:
    record = read(root, base)
    record["settings"]["attention_backend"] = "mha"
    record["created_utc"] = stamp
    if change:
        change(record)
    seed = record["settings"]["seed"]
    key = f"training/{stamp}_training_science-v1_mha_seed{seed}_48e95f8.json"
    write(root, key, record)
    return key


def test_legitimate_auto_mha_duplicate_counts_once(records: Path) -> None:
    duplicate = add_duplicate(records)
    assert problems(records, strict=True) == []
    rows = protocol_rows(load_store(ROOT, records), "science-v1", ["sdpa_v1_eval_cpu"])
    for backend in ("mha", "sdpa"):
        assert sorted(r["seed"] for r in rows if r["backend"] == backend) == [42, 43, 44]
    assert next(r for r in rows if r["backend"] == "mha" and r["seed"] == 42)["replicates"] == [duplicate]


def test_auto_mha_duplicate_with_contradictory_fingerprint_fails(records: Path) -> None:
    add_duplicate(records, change=lambda r: r["results"].update(best_model_identity_sha256="b" * 64))
    found = problems(records)
    assert any("sources disagree" in p for p in found) and any("conflicting training records" in p for p in found)
    with pytest.raises(RecordError):
        protocol_rows(load_store(ROOT, records), "science-v1", ["sdpa_v1_eval_cpu"])


def test_duplicate_with_conflicting_checkpoint_identity_fails(records: Path) -> None:
    add_duplicate(records, change=lambda r: r["results"].update(best_checkpoint_sha256="c" * 64))
    assert any("conflicting training records" in p and "different checkpoint files" in p for p in problems(records))


def test_duplicate_with_contradictory_losses_fails(records: Path) -> None:
    add_duplicate(records, change=lambda r: r["results"]["metrics"][-1].update(validation_loss=2.5))
    assert any("conflicting training records" in p and "losses" in p for p in problems(records))


def test_duplicate_with_contradictory_configuration_fails(records: Path) -> None:
    add_duplicate(records, change=lambda r: r["settings"].update(training_command=r["settings"]["training_command"].replace("--steps 1000", "--steps 2000")))
    assert any("conflicting training records" in p and "configuration differs" in p for p in problems(records))


def test_explicit_valid_supersession_resolves_a_conflict(records: Path) -> None:
    duplicate = add_duplicate(records, change=lambda r: r["results"]["metrics"][-1].update(validation_loss=2.5))
    index = json.loads((records / "SUPERSEDED.json").read_text())
    index["records"][duplicate] = {"reason": "test: conflicting re-record", "superseded_by": V1_42}
    (records / "SUPERSEDED.json").write_text(json.dumps(index))
    assert problems(records, strict=True) == []


@pytest.mark.parametrize("target", ["missing", "wrong_seed", "cycle", "other_kind"])
def test_invalid_supersession_fails(records: Path, target: str) -> None:
    index = json.loads((records / "SUPERSEDED.json").read_text())
    duplicate = add_duplicate(records, change=lambda r: r["results"]["metrics"][-1].update(validation_loss=2.5))
    destination = {"missing": "training/does_not_exist.json", "wrong_seed": V1_43, "cycle": duplicate, "other_kind": KV}[target]
    index["records"][duplicate] = {"reason": "test", "superseded_by": destination}
    if target == "cycle":
        second = add_duplicate(records, stamp="20261022T070000Z")
        index["records"][duplicate]["superseded_by"] = second
        index["records"][second] = {"reason": "test", "superseded_by": duplicate}
    (records / "SUPERSEDED.json").write_text(json.dumps(index))
    assert any("SUPERSEDED.json" in p for p in problems(records))


# ================================================================ Fix 3: KV speedup derivation


def kv_problems(root: Path) -> list[str]:
    return [p for p in problems(root) if p.startswith(("kv_cache/", "legacy/"))]


def test_stored_speedup_of_999_fails_and_is_never_rendered(records: Path) -> None:
    edit(records, KV, lambda r: r["results"]["grid"][0].update(speedup_cached_vs_uncached_sdpa=999.0))
    assert any("stored speedup_cached_vs_uncached_sdpa 999.0 differs from derived" in p for p in kv_problems(records))
    import importlib.util

    spec = importlib.util.spec_from_file_location("report", ROOT / "scripts/build_experiment_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    table = report.kv_table("cpu/float32", read(records, KV))
    assert "999.00×" not in table


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -2.0])
def test_non_finite_or_impossible_stored_ratio_fails(records: Path, value: float) -> None:
    edit(records, KV, lambda r: r["results"]["grid"][1].update(speedup_cached_vs_uncached_mha=value))
    assert any("speedup_cached_vs_uncached_mha" in p for p in kv_problems(records))


def test_altered_median_with_original_raw_samples_fails(records: Path) -> None:
    edit(records, KV, lambda r: r["results"]["grid"][3]["uncached_sdpa"]["total_ms"].update(median=1.0))
    assert any("summaries differ from raw samples" in p for p in kv_problems(records))


def test_altered_raw_sample_with_original_summary_fails(records: Path) -> None:
    edit(records, KV, lambda r: r["results"]["grid"][3]["cached"]["raw"]["step_seconds"][2].__setitem__(5, 0.5))
    assert any("summaries differ from raw samples" in p for p in kv_problems(records))


@pytest.mark.parametrize("value", [0.0, -0.001, float("nan")])
def test_zero_or_invalid_timing_denominator_fails_without_crashing(records: Path, value: float) -> None:
    def zero(record: dict) -> None:
        for steps in record["results"]["grid"][0]["cached"]["raw"]["step_seconds"]:
            for index in range(len(steps)):
                steps[index] = value
    edit(records, KV, zero)
    assert any("non-positive or non-finite" in p for p in kv_problems(records))


def test_valid_raw_samples_and_derived_ratios_pass() -> None:
    assert kv_problems(RECORDS) == []
    row = read(RECORDS, KV)["results"]["grid"][4]
    assert kv_speedups(row)["speedup_cached_vs_uncached_sdpa"] == row["speedup_cached_vs_uncached_sdpa"]


def test_historical_summary_only_record_passes_under_legacy_rules(records: Path) -> None:
    assert "raw" not in read(RECORDS, HISTORICAL_KV)["results"]["grid"][0]["cached"]
    assert kv_problems(records) == []
    edit(records, HISTORICAL_KV, lambda r: r["results"]["grid"][0].update(speedup_cached_vs_uncached_sdpa=999.0))
    assert any(HISTORICAL_KV in p and "differs from derived" in p for p in kv_problems(records))


def test_legacy_single_prompt_kv_benchmark_is_derived_from_samples(records: Path) -> None:
    edit(records, LEGACY_KV, lambda r: r["results"].update(speedup=9.99))
    assert any("stored speedup 9.99 differs from derived" in p for p in kv_problems(records))
