"""Experiment-record loading, identity, selection and provenance validation.

The rules are documented in docs/RECORD_SCHEMA.md. Both scripts/validate_records.py
and scripts/build_experiment_report.py use this module, so a record set that fails
validation cannot be reported.
"""

import hashlib
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from .identity import identity_config
from .model import canonical_backend
from .provenance import summarize

NAME = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_(?P<name>.+)_(?P<commit>[0-9a-f]{7})\.json$")
ENVIRONMENT_FIELDS = ("python", "torch", "platform", "processor", "device", "torch_threads", "git", "command")
PROTOCOLS = {
    "science-v1": {"manifest": "data/science/manifest.json", "data_dir": "data/science"},
    "science-v2": {"manifest": "data/science-v2/manifest.json", "data_dir": "data/science-v2"},
}
# train_tiny.py defaults (unchanged since 1fb8f37); used only to read records that
# predate explicit hyperparameters (record_version 1). Guarded by a test.
LEGACY_TRAIN_DEFAULTS = {"steps": 100, "batch_size": 8, "gradient_accumulation_steps": 1, "eval_interval": 25, "device": "auto"}
LEGACY_FLAGS = {"--steps": "steps", "--batch-size": "batch_size", "--gradient-accumulation-steps": "gradient_accumulation_steps",
                "--eval-interval": "eval_interval", "--device": "device"}


class RecordError(ValueError):
    """Raised when records cannot be reported without an unresolved problem."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_digest(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


@dataclass
class Store:
    repo_root: Path
    records_root: Path
    records: dict[str, dict]
    superseded: dict[str, dict]
    _file_hashes: dict = field(default_factory=dict)
    _commits: dict = field(default_factory=dict)

    def current(self, directory: str) -> list[tuple[str, dict]]:
        """(key, record) in time order, excluding superseded records."""
        return [(k, r) for k, r in sorted(self.records.items()) if k.startswith(directory + "/") and k not in self.superseded]

    def file_sha256(self, path: str) -> str | None:
        target = Path(path) if Path(path).is_absolute() else self.repo_root / path
        if target not in self._file_hashes:
            self._file_hashes[target] = sha256_bytes(target.read_bytes()) if target.is_file() else None
        return self._file_hashes[target]

    def commit_exists(self, ref: str) -> bool:
        if ref not in self._commits:
            result = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], check=False, cwd=self.repo_root, capture_output=True)
            self._commits[ref] = result.returncode == 0 and bool(re.fullmatch(r"[0-9a-f]{7,40}", ref))
        return self._commits[ref]


def load_store(repo_root: Path, records_root: Path | None = None) -> Store:
    records_root = records_root or repo_root / "experiments/records"
    records = {}
    for path in sorted(records_root.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = str(path.relative_to(records_root))
        record["_key"] = key
        record["_path"] = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
        records[key] = record
    index = records_root / "SUPERSEDED.json"
    superseded = json.loads(index.read_text())["records"] if index.exists() else {}
    return Store(repo_root, records_root, records, superseded)


# ---------------------------------------------------------------- datasets


def protocol_files(store: Store, protocol: str) -> dict[str, str]:
    """Published file hashes for a protocol, read from its committed manifest."""
    spec = PROTOCOLS[protocol]
    manifest = json.loads((store.repo_root / spec["manifest"]).read_text())
    if protocol == "science-v1":
        return {f"{spec['data_dir']}/train.txt": manifest["train_sha256"], f"{spec['data_dir']}/validation.txt": manifest["validation_sha256"]}
    return {f"{spec['data_dir']}/{name}": info["sha256"] for name, info in manifest["files"].items()}


def relative_data_path(store: Store, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        # Older records stored absolute paths; resolve them against this repository.
        parts = candidate.parts
        for index in range(len(parts)):
            if parts[index] == "data":
                return str(Path(*parts[index:]))
    return str(candidate)


def check_split(store: Store, protocol: str, label: str, info: dict, problems: list[str], where: str) -> None:
    path = relative_data_path(store, info["path"])
    if not path.startswith(PROTOCOLS[protocol]["data_dir"] + "/"):
        problems.append(f"{where}: {label} file {path} is outside the {protocol} corpus")
        return
    actual = store.file_sha256(path)
    if actual is None:
        problems.append(f"{where}: {label} file {path} does not exist")
    elif actual != info["sha256"]:
        problems.append(f"{where}: {label} hash {info['sha256'][:12]} differs from {path} ({actual[:12]})")
    published = protocol_files(store, protocol).get(path)
    if published is not None and published != info["sha256"]:
        problems.append(f"{where}: {label} hash differs from the {protocol} manifest")


# ---------------------------------------------------------------- training identity


class TrainingIdentity(NamedTuple):
    protocol: str
    backend: str
    seed: int
    config_sha256: str
    dataset_sha256: str


def training_hyperparameters(record: dict) -> dict:
    summary = record["results"].get("run_summary") or {}
    if "hyperparameters" in summary:
        return summary["hyperparameters"]
    # Legacy rule (record_version 1): parse the recorded command with train_tiny defaults.
    values = dict(LEGACY_TRAIN_DEFAULTS)
    tokens = shlex.split(record["settings"]["training_command"])
    for flag, name in LEGACY_FLAGS.items():
        if flag in tokens:
            raw = tokens[tokens.index(flag) + 1]
            values[name] = raw if name == "device" else int(raw)
    if values["device"] == "auto" and summary.get("device"):
        values["device"] = summary["device"]
    return {**values, "learning_rate": 3e-4, "optimizer": "AdamW"}


def training_identity(record: dict) -> TrainingIdentity:
    settings = record["settings"]
    config = {
        "model": identity_config(settings["model_config"]),
        "tokenizer": settings.get("tokenizer", "byte"),
        "hyperparameters": training_hyperparameters(record),
    }
    dataset = {split: settings["data"][split]["sha256"] for split in ("train", "validation")}
    return TrainingIdentity(settings["protocol"], canonical_backend(settings.get("attention_backend")), settings["seed"],
                            canonical_json_digest(config), canonical_json_digest(dataset))


def training_outcome(record: dict) -> tuple:
    results = record["results"]
    metrics = tuple((m["step"], m["validation_loss"], m["train_loss"]) for m in results["metrics"])
    return metrics, results["best_step"]


def check_training(store: Store, key: str, record: dict, problems: list[str]) -> None:
    settings = record["settings"]
    protocol = settings.get("protocol")
    if protocol not in PROTOCOLS:
        problems.append(f"{key}: unknown or missing protocol {protocol!r}")
        return
    if not isinstance(settings.get("seed"), int):
        problems.append(f"{key}: training record without integer seed")
    published = protocol_files(store, protocol)
    for split in ("train", "validation"):
        info = settings["data"][split]
        check_split(store, protocol, split, info, problems, key)
        if info["sha256"] not in published.values():
            problems.append(f"{key}: {split} data is not part of the {protocol} corpus")
    if "manifest_path" in settings["data"]:
        manifest = settings["data"]["manifest_path"]
        if manifest != PROTOCOLS[protocol]["manifest"]:
            problems.append(f"{key}: manifest {manifest} does not belong to {protocol}")
        elif store.file_sha256(manifest) != settings["data"].get("manifest_sha256"):
            problems.append(f"{key}: manifest hash does not match {manifest}")
    commit = settings.get("training_code_commit")
    if not commit or not store.commit_exists(commit):
        problems.append(f"{key}: training_code_commit {commit!r} is not in the repository")
    if record.get("record_version", 1) >= 2 and not record["results"].get("best_model_identity_sha256"):
        problems.append(f"{key}: schema v2 training record without model identity")
    if "best_checkpoint_sha256" not in record["results"]:
        problems.append(f"{key}: no checkpoint hash")


def select_training(store: Store, problems: list[str]) -> dict[TrainingIdentity, dict]:
    """One canonical record per experiment identity (rules in docs/RECORD_SCHEMA.md)."""
    groups: dict[TrainingIdentity, list[tuple[str, dict]]] = {}
    for key, record in store.current("training"):
        before = len(problems)
        check_training(store, key, record, problems)
        if len(problems) == before:
            groups.setdefault(training_identity(record), []).append((key, record))
    selected = {}
    for identity, members in groups.items():
        members.sort(key=lambda item: (item[1]["created_utc"], item[0]))
        canonical_key, canonical = members[0]
        conflicts = []
        for key, record in members[1:]:
            same_outcome = training_outcome(record) == training_outcome(canonical)
            a, b = record["results"].get("best_model_identity_sha256"), canonical["results"].get("best_model_identity_sha256")
            if not same_outcome or (a and b and a != b):
                conflicts.append(key)
        if conflicts:
            problems.append(f"conflicting training records for {identity.protocol}/{identity.backend}/seed {identity.seed}: "
                            f"{canonical_key} vs {', '.join(conflicts)} (supersede one explicitly)")
            continue
        canonical["_replicates"] = [key for key, _ in members[1:]]
        canonical["_identity"] = identity
        selected[identity] = canonical
    return selected


# ---------------------------------------------------------------- evaluation provenance


def attested_identities(store: Store, problems: list[str]) -> dict[str, str]:
    """checkpoint file SHA-256 -> model identity, from current attestation records."""
    attested: dict[str, str] = {}
    for key, record in store.current("identity"):
        for row in record["results"]["attested"]:
            training = store.records.get(row["training_record"])
            if training is None or row["training_record"] in store.superseded:
                problems.append(f"{key}: attests {row['training_record']}, which is not a current training record")
                continue
            if training["results"].get(f"{row['kind']}_checkpoint_sha256") != row["checkpoint_sha256"]:
                problems.append(f"{key}: {row['kind']} checkpoint hash differs from {row['training_record']}")
            previous = attested.setdefault(row["checkpoint_sha256"], row["model_identity_sha256"])
            if previous != row["model_identity_sha256"]:
                problems.append(f"{key}: checkpoint {row['checkpoint_sha256'][:12]} attested with two model identities")
        for row in record["results"].get("regeneration_checks", []):
            original = attested.get(row["original_checkpoint_sha256"])
            if row["model_identity_equal"] != (original == row["model_identity_sha256"]):
                problems.append(f"{key}: regeneration check for {row['training_record']} is inconsistent with its attestation")
    return attested


def model_identity_of(store: Store, training: dict) -> str | None:
    own = training["results"].get("best_model_identity_sha256")
    return own or attested_identities(store, []).get(training["results"].get("best_checkpoint_sha256"))


def training_by_checkpoint(store: Store) -> dict[str, list[tuple[str, dict]]]:
    index: dict[str, list[tuple[str, dict]]] = {}
    for key, record in store.current("training"):
        index.setdefault(record["results"].get("best_checkpoint_sha256"), []).append((key, record))
    return index


def resolve_row(store: Store, where: str, row: dict, protocol: str | None, problems: list[str], index: dict) -> tuple[str, dict] | None:
    matches = index.get(row.get("checkpoint_sha256"), [])
    identities = set()
    for _, candidate in matches:
        try:
            identities.add(training_identity(candidate))
        except (KeyError, TypeError):
            identities.add(None)
    # Several records may describe one checkpoint only as replicates of a single
    # experiment identity (e.g. a legacy "auto" record and its "mha" re-record).
    if not matches or len(identities) != 1 or None in identities:
        problems.append(f"{where}: checkpoint {str(row.get('checkpoint_sha256'))[:12]} matches {len(matches)} current training records "
                        f"with {len(identities)} experiment identities (expected exactly one)")
        return None
    key, training = min(matches, key=lambda item: (item[1]["created_utc"], item[0]))
    settings = training["settings"]
    if protocol is not None and settings.get("protocol") != protocol:
        problems.append(f"{where}: protocol {protocol} but training record {key} is {settings.get('protocol')}")
    if "seed" in row and row["seed"] != settings.get("seed"):
        problems.append(f"{where}: seed {row.get('seed')} but training record {key} has seed {settings.get('seed')}")
    backend = row.get("attention_backend", row.get("backend"))
    if "attention_backend" in row and canonical_backend(backend) != canonical_backend(settings.get("attention_backend")):
        problems.append(f"{where}: backend {backend} but training record {key} used {settings.get('attention_backend')}")
    if "selected_step" in row and row["selected_step"] != training["results"]["best_step"]:
        problems.append(f"{where}: selected step {row['selected_step']} but training record {key} selected {training['results']['best_step']}")
    if "corpus_sha256" in row and row["corpus_sha256"] != settings["data"].get("corpus_sha256_in_checkpoint"):
        problems.append(f"{where}: corpus identity differs from training record {key}")
    a, b = row.get("model_identity_sha256"), model_identity_of(store, training)
    if a and b and a != b:
        problems.append(f"{where}: model identity differs from training record {key}")
    return key, training


def check_evaluation(store: Store, key: str, record: dict, problems: list[str], index: dict) -> list[tuple[dict, str]]:
    settings = record["settings"]
    protocol = settings.get("protocol")
    if protocol not in PROTOCOLS:
        problems.append(f"{key}: unknown or missing protocol {protocol!r}")
        return []
    for name, info in settings["splits"].items():
        check_split(store, protocol, f"split {name}", info, problems, key)
    resolved, seen = [], {}
    for position, row in enumerate(record["results"]):
        where = f"{key} row {position}"
        if record.get("record_version", 1) >= 2 and not row.get("model_identity_sha256"):
            problems.append(f"{where}: schema v2 evaluation row without model identity")
        match = resolve_row(store, where, row, protocol, problems, index)
        if match is None:
            continue
        training_key, training = match
        validation = settings["splits"].get("validation")
        if validation and validation["sha256"] != training["settings"]["data"]["validation"]["sha256"]:
            problems.append(f"{where}: validation split differs from the one training record {training_key} selected on")
        identity = training_identity(training)
        if identity in seen:
            problems.append(f"{where}: duplicates row {seen[identity]} for the same experiment identity")
        seen[identity] = position
        resolved.append((row, training_key))
    return resolved


# ---------------------------------------------------------------- benchmark raw samples


def kv_summary_from_raw(raw: dict, batch: int, new_tokens: int) -> dict:
    """KV-cache mode summaries recomputed from per-repeat, per-step seconds."""
    runs = raw["step_seconds"]
    totals = [sum(steps) for steps in runs]
    return {
        "first_token_ms": summarize([1000 * steps[0] for steps in runs]),
        "decode_ms_per_step": summarize([1000 * t for steps in runs for t in steps[1:]]),
        "total_ms": summarize([1000 * t for t in totals]),
        "tokens_per_second": summarize([batch * new_tokens / t for t in totals]),
        "median_step_ms_by_index": [1000 * sorted(steps[i] for steps in runs)[len(runs) // 2] for i in range(new_tokens)],
    }


def _close(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def check_benchmark(key: str, record: dict, problems: list[str]) -> None:
    experiment, settings = record["experiment"], record["settings"]
    for name in ("warmup", "repeats", "seed"):
        if name not in settings:
            problems.append(f"{key}: benchmark settings lack {name}")
    if experiment == "kv_cache_scaling":
        if "raw" not in record["results"]["grid"][0]["cached"]:
            problems.append(f"{key}: no raw per-repeat timing samples (historical records must be superseded)")
            return
        scenarios = [(row, row["batch"], settings["new_tokens"]) for row in record["results"]["grid"]]
        window = record["results"]["window"]
        scenarios.append((window, window["batch"], window["new_tokens"]))
        for row, batch, new_tokens in scenarios:
            for mode in settings["modes"]:
                raw = row[mode]["raw"]
                if len(raw["step_seconds"]) != settings["repeats"] or any(len(s) != new_tokens for s in raw["step_seconds"]):
                    problems.append(f"{key}: raw sample count does not match repeats x tokens ({mode}, B={batch})")
                    continue
                recomputed = kv_summary_from_raw(raw, batch, new_tokens)
                if not all(_close(recomputed[name], row[mode][name]) for name in recomputed):
                    problems.append(f"{key}: {mode} B={batch} summaries differ from raw samples")
    elif experiment == "attention_backends":
        for row in record["results"]:
            runs = row["samples_seconds"]
            if len(runs) != settings["repeats"] or not _close(summarize([1000 * r["forward"] for r in runs]), row["forward_ms"]):
                problems.append(f"{key}: {row['backend']} B={row['batch']} L={row['seq_len']} forward summary differs from raw samples")
            if "prefill_ms" in row:
                decode = summarize([1000 * s for r in runs for s in r["decode_steps"]])
                if not (_close(summarize([1000 * r["prefill"] for r in runs]), row["prefill_ms"]) and _close(decode, row["decode_ms_per_step"])):
                    problems.append(f"{key}: {row['backend']} B={row['batch']} L={row['seq_len']} prefill/decode summary differs from raw samples")
    elif experiment == "training_step":
        for row in record["results"]:
            samples = row["step_seconds_samples"]
            tokens = row["batch"] * row["seq_len"]
            if len(samples) != settings["repeats"] or not (_close(summarize([1000 * s for s in samples]), row["step_ms"])
                                                         and _close(summarize([tokens / s for s in samples]), row["tokens_per_second"])):
                problems.append(f"{key}: {row['backend']} L={row['seq_len']} summary differs from raw samples")


# ---------------------------------------------------------------- whole-store validation


def validate_store(store: Store) -> list[str]:
    problems: list[str] = []
    for key, entry in store.superseded.items():
        for path in (key, entry["superseded_by"]):
            if path not in store.records:
                problems.append(f"SUPERSEDED.json names a missing record: {path}")
        if entry["superseded_by"] == key:
            problems.append(f"SUPERSEDED.json: {key} supersedes itself")
    for key, record in sorted(store.records.items()):
        match = NAME.match(Path(key).name)
        if not match:
            problems.append(f"{key}: unexpected file name")
            continue
        for name in ("record_version", "created_utc", "experiment", "environment", "settings", "results"):
            if name not in record:
                problems.append(f"{key}: missing {name}")
        environment = record.get("environment", {})
        missing = [name for name in ENVIRONMENT_FIELDS if name not in environment]
        if missing:
            problems.append(f"{key}: environment lacks {missing}")
            continue
        git = environment["git"]
        if record["created_utc"] != match["stamp"] or not str(git["commit"]).startswith(match["commit"]):
            problems.append(f"{key}: file name does not match created_utc/commit")
        if not store.commit_exists(str(git["commit"])):
            problems.append(f"{key}: commit {git['commit']} not in repository")
        if git["dirty"] and key not in store.superseded:
            problems.append(f"{key}: dirty working tree but not listed in SUPERSEDED.json")
        if key not in store.superseded and record["experiment"] in {"attention_backends", "kv_cache_scaling", "training_step"}:
            check_benchmark(key, record, problems)
    select_training(store, problems)
    attested_identities(store, problems)
    index = training_by_checkpoint(store)
    for key, record in store.current("evaluation"):
        check_evaluation(store, key, record, problems, index)
    for key, record in store.current("attention_quality"):
        for position, row in enumerate(record["results"]):
            resolve_row(store, f"{key} row {position}", row, None, problems, index)
    for key, record in store.current("legacy"):
        settings = record["settings"]
        resolve_row(store, key, {"checkpoint_sha256": settings.get("checkpoint_sha256")}, None, problems, index)
        if "validation_text" in settings:
            path = relative_data_path(store, settings["validation_text"])
            if store.file_sha256(path) != settings["validation_sha256"]:
                problems.append(f"{key}: validation text hash differs from {path}")
    return problems


def protocol_rows(store: Store, protocol: str, evaluation_names: list[str]) -> list[dict]:
    """Training identities of a protocol joined to named evaluations by checkpoint identity.

    Fails closed: raises RecordError on any validation problem, on an evaluation
    row that does not resolve, or on a training identity missing from an evaluation.
    """
    problems = validate_store(store)
    if problems:
        raise RecordError("records failed validation:\n" + "\n".join(problems))
    selected = {identity: record for identity, record in select_training(store, []).items() if identity.protocol == protocol}
    index = training_by_checkpoint(store)
    evaluations = {}
    for key, record in store.current("evaluation"):
        name = NAME.match(Path(key).name)["name"]
        if name in evaluation_names:
            evaluations[name] = (key, record)  # latest current record with this name
    rows = {}
    for identity, record in selected.items():
        summary = record["results"]["run_summary"]
        rows[identity] = {"seed": identity.seed, "backend": identity.backend, "best_step": record["results"]["best_step"],
                          "tokens_per_second": summary["train_tokens_per_second"], "train_seconds": summary["train_seconds"],
                          "peak_rss_mib": summary["process_peak_rss_bytes"] / 2**20, "record": record["_path"],
                          "replicates": record["_replicates"]}
    for name in evaluation_names:
        if name not in evaluations:
            continue
        key, record = evaluations[name]
        for row, training_key in check_evaluation(store, key, record, [], index):
            identity = training_identity(store.records[training_key])
            if identity not in rows:
                raise RecordError(f"{key}: evaluates {training_key}, which is not a selected {protocol} training identity")
            rows[identity][name] = row["splits"]
        missing = [identity for identity in rows if name not in rows[identity]]
        if missing:
            raise RecordError(f"{key}: no evaluation row for " + ", ".join(f"{i.backend}/seed {i.seed}" for i in missing))
    return sorted(rows.values(), key=lambda r: (r["seed"], r["backend"]))
