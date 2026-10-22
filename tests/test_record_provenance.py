"""Findings 2, 3 and 4: experiment identity, evaluation provenance and raw benchmark samples.

Every test works on a copy of experiments/records so the committed records stay untouched.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from minilm_lab.records import (
    LEGACY_TRAIN_DEFAULTS,
    RecordError,
    load_store,
    protocol_rows,
    validate_store,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"
V1_MHA_42 = "training/20261022T041654Z_training_science-v1_auto_seed42_48e95f8.json"
V2_FINAL = "evaluation/20261022T043652Z_v2_final_cpu_897db37.json"
V1_EVAL = "evaluation/20261022T041615Z_sdpa_v1_eval_cpu_48e95f8.json"


@pytest.fixture
def records(tmp_path: Path) -> Path:
    copy = tmp_path / "records"
    shutil.copytree(RECORDS, copy)
    return copy


def read(root: Path, key: str) -> dict:
    return json.loads((root / key).read_text())


def write(root: Path, key: str, record: dict) -> None:
    (root / key).write_text(json.dumps(record, indent=2))


def problems(root: Path) -> list[str]:
    return validate_store(load_store(ROOT, root))


def add_mha_duplicate_of_auto_seed42(root: Path, stamp: str = "20261022T060000Z", **metric_change) -> str:
    record = read(root, V1_MHA_42)
    record["settings"]["attention_backend"] = "mha"  # canonical name for the legacy "auto"
    record["created_utc"] = stamp
    if metric_change:
        record["results"]["metrics"][-1].update(metric_change)
    key = f"training/{stamp}_training_science-v1_mha_seed42_48e95f8.json"
    write(root, key, record)
    return key


def test_committed_records_validate() -> None:
    assert problems(RECORDS) == []


# ---------------------------------------------------------------- finding 2


def test_legacy_auto_and_canonical_mha_seed42_count_once(records: Path) -> None:
    duplicate = add_mha_duplicate_of_auto_seed42(records)
    assert problems(records) == []
    rows = protocol_rows(load_store(ROOT, records), "science-v1", ["sdpa_v1_eval_cpu"])
    mha = [row for row in rows if row["backend"] == "mha"]
    assert sorted(row["seed"] for row in mha) == [42, 43, 44]  # not inflated to four
    assert next(row for row in mha if row["seed"] == 42)["replicates"] == [duplicate]
    assert sorted(row["seed"] for row in rows if row["backend"] == "sdpa") == [42, 43, 44]  # all valid seeds kept


def test_conflicting_duplicate_identity_fails_validation_and_reporting(records: Path) -> None:
    add_mha_duplicate_of_auto_seed42(records, validation_loss=2.5)
    assert any("conflicting training records" in p and "seed 42" in p for p in problems(records))
    with pytest.raises(RecordError, match="conflicting training records"):
        protocol_rows(load_store(ROOT, records), "science-v1", ["sdpa_v1_eval_cpu"])


def test_explicit_supersession_resolves_a_conflict_and_keeps_the_audit_trail(records: Path) -> None:
    duplicate = add_mha_duplicate_of_auto_seed42(records, validation_loss=2.5)
    index = json.loads((records / "SUPERSEDED.json").read_text())
    index["records"][duplicate] = {"reason": "test: conflicting re-record", "superseded_by": V1_MHA_42}
    (records / "SUPERSEDED.json").write_text(json.dumps(index))
    assert problems(records) == []
    assert duplicate in load_store(ROOT, records).records  # kept on disk, not discarded


def test_historical_report_rows_are_reproduced() -> None:
    store = load_store(ROOT, RECORDS)
    v1 = protocol_rows(store, "science-v1", ["sdpa_v1_eval_cpu"])
    perplexity = {(r["backend"], r["seed"]): round(r["sdpa_v1_eval_cpu"]["validation"]["perplexity"], 3) for r in v1}
    assert perplexity == {("mha", 42): 17.583, ("mha", 43): 18.989, ("mha", 44): 21.068,
                          ("sdpa", 42): 17.683, ("sdpa", 43): 19.009, ("sdpa", 44): 20.416}
    v2 = protocol_rows(store, "science-v2", ["v2_final_cpu"])
    assert len(v2) == 6 and all("test" in row["v2_final_cpu"] for row in v2)


# ---------------------------------------------------------------- finding 3


def mutate_evaluation(root: Path, key: str, change) -> None:
    record = read(root, key)
    change(record)
    write(root, key, record)


def test_zeroed_checkpoint_hash_is_rejected(records: Path) -> None:
    mutate_evaluation(records, V2_FINAL, lambda r: r["results"][0].update(checkpoint_sha256="0" * 64))
    assert any("matches 0 current training records" in p for p in problems(records))
    with pytest.raises(RecordError):
        protocol_rows(load_store(ROOT, records), "science-v2", ["v2_final_cpu"])


def test_evaluation_associated_with_wrong_training_run_is_rejected(records: Path) -> None:
    def swap(record: dict) -> None:
        a, b = record["results"][0], record["results"][2]  # seeds 42 and 43 of the same backend
        a["checkpoint_sha256"], b["checkpoint_sha256"] = b["checkpoint_sha256"], a["checkpoint_sha256"]
    mutate_evaluation(records, V2_FINAL, swap)
    found = problems(records)
    assert any("seed 42 but training record" in p for p in found) and any("seed 43 but training record" in p for p in found)


def test_changed_evaluation_corpus_identity_is_rejected(records: Path) -> None:
    mutate_evaluation(records, V2_FINAL, lambda r: r["settings"]["splits"]["test"].update(sha256="f" * 64))
    assert any("split test hash" in p for p in problems(records))
    mutate_evaluation(records, V2_FINAL, lambda r: r["settings"]["splits"]["test"].update(sha256=read(RECORDS, V2_FINAL)["settings"]["splits"]["test"]["sha256"]))
    mutate_evaluation(records, V2_FINAL, lambda r: r["results"][1].update(corpus_sha256="e" * 64))
    assert any("corpus identity differs" in p for p in problems(records))


def test_evaluation_split_from_another_protocol_is_rejected(records: Path) -> None:
    v1_validation = read(RECORDS, V1_EVAL)["settings"]["splits"]["validation"]
    mutate_evaluation(records, V2_FINAL, lambda r: r["settings"]["splits"].update(validation=v1_validation))
    found = problems(records)
    assert any("outside the science-v2 corpus" in p for p in found)


def test_changed_protocol_is_rejected(records: Path) -> None:
    mutate_evaluation(records, V2_FINAL, lambda r: r["settings"].update(protocol="science-v1"))
    found = problems(records)
    assert any("protocol science-v1 but training record" in p for p in found)
    mutate_evaluation(records, V2_FINAL, lambda r: r["settings"].update(protocol="science-v9"))
    assert any("unknown or missing protocol 'science-v9'" in p for p in problems(records))


def test_contradictory_duplicate_training_records_are_rejected(records: Path) -> None:
    add_mha_duplicate_of_auto_seed42(records, stamp="20261022T070000Z", validation_loss=3.0)
    assert any("conflicting training records" in p for p in problems(records))


def test_invalid_commit_and_manifest_references_are_rejected(records: Path) -> None:
    record = read(records, V1_MHA_42)
    record["settings"]["training_code_commit"] = "deadbee"
    write(records, V1_MHA_42, record)
    assert any("training_code_commit 'deadbee' is not in the repository" in p for p in problems(records))
    record["settings"]["training_code_commit"] = "48e95f8"
    record["settings"]["data"]["manifest_sha256"] = "0" * 64
    write(records, V1_MHA_42, record)
    assert any("manifest hash does not match" in p for p in problems(records))
    record["settings"]["data"]["manifest_path"] = "data/science-v2/manifest.json"
    write(records, V1_MHA_42, record)
    assert any("does not belong to science-v1" in p for p in problems(records))


def test_invalid_environment_commit_is_rejected(records: Path) -> None:
    record = read(records, V2_FINAL)
    record["environment"]["git"]["commit"] = "897db37" + "0" * 33
    write(records, V2_FINAL, record)
    assert any("not in repository" in p for p in problems(records))


def test_model_identity_mismatch_is_rejected(records: Path) -> None:
    training = read(records, V1_MHA_42)
    training["results"]["best_model_identity_sha256"] = "a" * 64
    write(records, V1_MHA_42, training)
    mutate_evaluation(records, V1_EVAL, lambda r: [row.update(model_identity_sha256="b" * 64) for row in r["results"] if row["seed"] == 42 and row["attention_backend"] == "auto"])
    # The forged declared value already contradicts the artifact/baseline identity.
    found = problems(records)
    assert any("model identity sources disagree" in p for p in found)


def test_training_data_outside_protocol_is_rejected(records: Path) -> None:
    record = read(records, V1_MHA_42)
    record["settings"]["data"]["train"] = {"path": "data/science-v2/train.txt", "sha256": read(RECORDS, V1_MHA_42)["settings"]["data"]["train"]["sha256"]}
    write(records, V1_MHA_42, record)
    assert any("outside the science-v1 corpus" in p for p in problems(records))


def test_legacy_hyperparameter_defaults_match_train_tiny() -> None:
    spec = importlib.util.spec_from_file_location("train_tiny", ROOT / "scripts/train_tiny.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = vars(module.build_parser().parse_args(["corpus.txt"]))
    assert {name: defaults[name] for name in LEGACY_TRAIN_DEFAULTS} == LEGACY_TRAIN_DEFAULTS


# ---------------------------------------------------------------- finding 4


def current_kv(root: Path) -> list[str]:
    store = load_store(ROOT, root)
    return [key for key, _ in store.current("kv_cache")]


def test_current_kv_records_have_raw_samples_and_historical_ones_are_marked() -> None:
    store = load_store(ROOT, RECORDS)
    for key, record in store.current("kv_cache"):
        assert "raw" in record["results"]["grid"][0]["cached"], key
    for key, record in store.records.items():
        if key.startswith("kv_cache/") and "raw" not in record["results"]["grid"][0]["cached"]:
            assert "raw" in store.superseded[key]["reason"]


def test_kv_summary_that_disagrees_with_raw_samples_is_rejected(records: Path) -> None:
    key = current_kv(records)[0]
    record = read(records, key)
    record["results"]["grid"][2]["uncached_sdpa"]["total_ms"]["p95"] *= 1.001
    write(records, key, record)
    assert any("summaries differ from raw samples" in p for p in problems(records))


def test_kv_record_without_raw_samples_is_rejected(records: Path) -> None:
    key = current_kv(records)[0]
    record = read(records, key)
    for row in record["results"]["grid"]:
        for mode in record["settings"]["modes"]:
            row[mode].pop("raw")
    write(records, key, record)
    assert any("no raw per-repeat timing samples" in p for p in problems(records))


# ---------------------------------------------------------------- exit codes


def test_validator_failure_propagates_through_exit_codes_and_pipelines(records: Path) -> None:
    mutate_evaluation(records, V2_FINAL, lambda r: r["results"][0].update(checkpoint_sha256="0" * 64))
    command = [sys.executable, str(ROOT / "scripts/validate_records.py"), "--records-dir", str(records)]
    assert subprocess.run(command, check=False, capture_output=True, cwd=ROOT).returncode == 1
    piped = subprocess.run(["bash", "-o", "pipefail", "-c", " ".join(command) + " | tail -n 1"], check=False, capture_output=True, cwd=ROOT)
    assert piped.returncode == 1
    clean = [sys.executable, str(ROOT / "scripts/validate_records.py")]
    assert subprocess.run(clean, check=False, capture_output=True, cwd=ROOT).returncode == 0


# ---------------------------------------------------------------- attested identities (finding 1 for historical runs)

ATTESTATION = "identity/20261022T053914Z_model_identity_attestation_935a229.json"


def test_attestation_with_wrong_checkpoint_hash_is_rejected(records: Path) -> None:
    record = read(records, ATTESTATION)
    record["results"]["attested"][0]["checkpoint_sha256"] = "0" * 64
    write(records, ATTESTATION, record)
    assert any("checkpoint hash differs" in p for p in problems(records))


def test_inconsistent_regeneration_check_is_rejected(records: Path) -> None:
    record = read(records, ATTESTATION)
    record["results"]["regeneration_checks"][0]["model_identity_sha256"] = "1" * 64
    write(records, ATTESTATION, record)
    assert any("regeneration check" in p for p in problems(records))


def test_attested_identity_is_enforced_for_historical_evaluations(records: Path) -> None:
    mutate_evaluation(records, V1_EVAL, lambda r: r["results"][0].update(model_identity_sha256="2" * 64))
    assert any("model identity differs" in p for p in problems(records))
