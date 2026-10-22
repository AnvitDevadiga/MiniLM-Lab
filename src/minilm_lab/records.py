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
# Model-identity evidence levels, strongest first. Only the first two count as verification.
ARTIFACT_VERIFIED = "artifact_verified"
BASELINE_VERIFIED = "trusted_baseline_verified"
METADATA_ONLY = "metadata_consistent_only"
UNVERIFIED = "unverified"
VERIFIED = (ARTIFACT_VERIFIED, BASELINE_VERIFIED)
# Audited baseline of model identities for historical checkpoints, built by
# scripts/build_identity_baseline.py from the original checkpoint files after checking
# their bytes against the recorded SHA-256. It is trusted only if its SHA-256 equals the
# value pinned here, so any change to it is a reviewable code change. This is an
# integrity anchor inside this repository, not a cryptographic signature.
BASELINE_PATH = "experiments/baselines/model_identity_baseline.json"
BASELINE_SHA256 = "02992d75cde737c37953cc7156cdbf08d43b910ad4347fae9d4bab178fdd2439"
LEGACY_FLAGS = {"--steps": "steps", "--batch-size": "batch_size", "--gradient-accumulation-steps": "gradient_accumulation_steps",
                "--eval-interval": "eval_interval", "--device": "device"}


class RecordError(ValueError):
    """Raised when records cannot be reported without an unresolved problem."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_digest(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


_ARTIFACT_CACHE: dict = {}


@dataclass
class Store:
    repo_root: Path
    records_root: Path
    records: dict[str, dict]
    superseded: dict[str, dict]
    artifact_root: Path | None = None  # where checkpoint artifacts are looked up (default: repo_root)
    baseline_path: Path | None = None  # None: no audited baseline available
    _file_hashes: dict = field(default_factory=dict)
    _commits: dict = field(default_factory=dict)
    _evidence: dict = field(default_factory=dict)
    _baseline: tuple | None = None

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


def load_store(repo_root: Path, records_root: Path | None = None, artifact_root: Path | None = None,
               baseline_path: Path | None | bool = True) -> Store:
    """baseline_path=True uses the committed baseline; None/False disables it (for tests)."""
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
    baseline = repo_root / BASELINE_PATH if baseline_path is True else (baseline_path or None)
    return Store(repo_root, records_root, records, superseded, artifact_root or repo_root, baseline)


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


def duplicate_conflicts(store: Store, first: tuple[str, dict], other: tuple[str, dict]) -> list[str]:
    """Reasons two records in the same experiment slot cannot be the same experiment."""
    (key_a, a), (key_b, b) = first, other
    reasons = []
    if training_identity(a).config_sha256 != training_identity(b).config_sha256:
        reasons.append("configuration differs")
    if training_outcome(a) != training_outcome(b):
        reasons.append("recorded losses or selected step differ")
    identity_a, identity_b = resolve_identity(store, key_a)[0], resolve_identity(store, key_b)[0]
    if identity_a.fingerprint is None or identity_b.fingerprint is None:
        if a["results"].get("best_checkpoint_sha256") != b["results"].get("best_checkpoint_sha256"):
            reasons.append("different checkpoint files without established model identities")
        if (identity_a.sources and identity_a.fingerprint is None) or (identity_b.sources and identity_b.fingerprint is None):
            reasons.append("model identity sources disagree")
    elif identity_a.fingerprint != identity_b.fingerprint:
        reasons.append("model identities differ")
    return reasons


def select_training(store: Store, problems: list[str]) -> dict[TrainingIdentity, dict]:
    """One record per experiment (rules in docs/RECORD_SCHEMA.md).

    Records are grouped by experiment slot (protocol, canonical backend, seed, dataset)
    after backend canonicalization. Members of a slot are duplicates of one experiment
    only if configuration, outcomes and resolved checkpoint identity all agree; then the
    earliest record represents them (they are identical, so the choice changes nothing).
    Any disagreement is a conflict: no record is selected and validation fails until an
    explicit supersession resolves it.
    """
    slots: dict[tuple, list[tuple[str, dict]]] = {}
    for key, record in store.current("training"):
        before = len(problems)
        check_training(store, key, record, problems)
        if len(problems) == before:
            identity = training_identity(record)
            slots.setdefault((identity.protocol, identity.backend, identity.seed, identity.dataset_sha256), []).append((key, record))
    selected = {}
    for (protocol, backend, seed, _), members in slots.items():
        members.sort(key=lambda item: (item[1]["created_utc"], item[0]))
        reasons = {key: duplicate_conflicts(store, members[0], (key, record)) for key, record in members[1:]}
        conflicts = {key: why for key, why in reasons.items() if why}
        if conflicts:
            detail = "; ".join(f"{key}: {', '.join(why)}" for key, why in conflicts.items())
            problems.append(f"conflicting training records for {protocol}/{backend}/seed {seed} ({members[0][0]} vs {detail}); "
                            "supersede one explicitly")
            continue
        canonical = members[0][1]
        canonical["_replicates"] = [key for key, _ in members[1:]]
        canonical["_identity"] = training_identity(canonical)
        selected[canonical["_identity"]] = canonical
    return selected


# ---------------------------------------------------------------- evaluation provenance


class IdentityEvidence(NamedTuple):
    fingerprint: str | None  # None when unknown or when sources disagree
    status: str  # one of ARTIFACT_VERIFIED, BASELINE_VERIFIED, METADATA_ONLY, UNVERIFIED
    sources: dict  # source name -> fingerprint, as found


def load_baseline(store: Store, problems: list[str]) -> dict:
    """Audited baseline entries keyed by checkpoint SHA-256, only if the file matches its pin."""
    if store._baseline is None:
        issues: list[str] = []
        entries: dict = {}
        path = store.baseline_path
        if path is not None and path.is_file():
            data = path.read_bytes()
            if sha256_bytes(data) != BASELINE_SHA256:
                issues.append(f"{BASELINE_PATH}: SHA-256 {sha256_bytes(data)[:12]} does not match the pinned audited baseline; not trusted")
            else:
                entries = json.loads(data)["entries"]
        store._baseline = (entries, issues)
    entries, issues = store._baseline
    problems.extend(issue for issue in issues if issue not in problems)
    return entries


def checkpoint_artifact(store: Store, record: dict, kind: str) -> Path | None:
    tokens = shlex.split(record["settings"].get("training_command", ""))
    if "--output-dir" not in tokens:
        return None
    path = store.artifact_root / tokens[tokens.index("--output-dir") + 1] / ("best_checkpoint.pt" if kind == "best" else "tiny_checkpoint.pt")
    return path if path.is_file() else None


def artifact_fingerprint(path: Path, expected_sha256: str) -> tuple[str, str | None]:
    """(file SHA-256, model identity recomputed from tensors if the bytes match the expectation)."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _ARTIFACT_CACHE:
        _ARTIFACT_CACHE[key] = [sha256_bytes(path.read_bytes()), None]
    entry = _ARTIFACT_CACHE[key]
    if entry[0] == expected_sha256 and entry[1] is None:
        import torch  # deferred: only needed when artifacts are present

        from .identity import checkpoint_identity

        entry[1] = checkpoint_identity(torch.load(path, map_location="cpu", weights_only=True))["sha256"]
    return entry[0], entry[1] if entry[0] == expected_sha256 else None


def resolve_identity(store: Store, key: str, kind: str = "best") -> tuple[IdentityEvidence, list[str]]:
    """Effective model identity of a training record's checkpoint and the evidence behind it.

    Sources: the record's declared fingerprint, attestation rows for the same checkpoint
    hash, the pinned audited baseline, and the checkpoint artifact itself (bytes checked
    against the recorded SHA-256, fingerprint recomputed from its tensors). All available
    sources must agree; a disagreement yields no fingerprint and a problem.
    """
    if (key, kind) in store._evidence:
        return store._evidence[(key, kind)]
    problems: list[str] = []
    record = store.records[key]
    expected = record["results"].get(f"{kind}_checkpoint_sha256")
    sources: dict[str, str] = {}
    declared = record["results"].get(f"{kind}_model_identity_sha256")
    if declared:
        sources["declared"] = declared
    attested = set()
    for attestation_key, attestation in store.current("identity"):
        for row in attestation["results"]["attested"]:
            if expected and row["checkpoint_sha256"] == expected:
                attested.add(row["model_identity_sha256"])
    if len(attested) > 1:
        problems.append(f"{key}: {kind} checkpoint attested with {len(attested)} different model identities")
    elif attested:
        sources["attested"] = attested.pop()
    baseline = load_baseline(store, problems).get(expected or "")
    if baseline:
        sources["baseline"] = baseline["model_identity_sha256"]
    artifact = checkpoint_artifact(store, record, kind) if expected else None
    if artifact is not None:
        file_sha, fingerprint = artifact_fingerprint(artifact, expected)
        if fingerprint is None:
            problems.append(f"{key}: {kind} checkpoint {artifact.name} bytes ({file_sha[:12]}) do not match the recorded SHA-256 ({expected[:12]})")
        else:
            sources["artifact"] = fingerprint
    values = set(sources.values())
    if len(values) > 1:
        detail = ", ".join(f"{name}={value[:12]}" for name, value in sorted(sources.items()))
        problems.append(f"{key}: {kind} model identity sources disagree ({detail})")
        evidence = IdentityEvidence(None, UNVERIFIED, sources)
    elif not values:
        evidence = IdentityEvidence(None, UNVERIFIED, sources)
    elif "artifact" in sources and not any("artifact" in p and "do not match" in p for p in problems):
        evidence = IdentityEvidence(values.pop(), ARTIFACT_VERIFIED, sources)
    elif "baseline" in sources:
        evidence = IdentityEvidence(values.pop(), BASELINE_VERIFIED, sources)
    else:
        evidence = IdentityEvidence(values.pop(), METADATA_ONLY, sources)
    if any("do not match the recorded" in p for p in problems):
        evidence = IdentityEvidence(None, UNVERIFIED, sources)
    store._evidence[(key, kind)] = (evidence, problems)
    return evidence, problems


def check_attestations(store: Store, problems: list[str]) -> None:
    """Attestation rows must point at current training records and their recorded hashes."""
    for key, record in store.current("identity"):
        for row in record["results"]["attested"]:
            training = store.records.get(row["training_record"])
            if training is None or row["training_record"] in store.superseded:
                problems.append(f"{key}: attests {row['training_record']}, which is not a current training record")
            elif training["results"].get(f"{row['kind']}_checkpoint_sha256") != row["checkpoint_sha256"]:
                problems.append(f"{key}: {row['kind']} checkpoint hash differs from {row['training_record']}")
        for row in record["results"].get("regeneration_checks", []):
            training = row["training_record"]
            if training not in store.records or training in store.superseded:
                problems.append(f"{key}: regeneration check for unknown training record {training}")
                continue
            evidence, _ = resolve_identity(store, training)
            if evidence.fingerprint is None or row["model_identity_equal"] != (evidence.fingerprint == row["model_identity_sha256"]):
                problems.append(f"{key}: regeneration check for {training} is inconsistent with its resolved identity")


def model_identity_of(store: Store, training: dict) -> str | None:
    return resolve_identity(store, training["_key"])[0].fingerprint


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


# Stored KV-cache ratios and how they are derived. Each is
#   median over repeats of the uncached mode's total generation time (ms)
#   ÷ median over repeats of the cached mode's total generation time (ms),
# where a repeat's total is the sum of its per-step seconds × 1000 (prefill included).
# Unitless; > 1 means the cache is faster. Records before the first release name the
# legacy MHA mode "uncached_default".
KV_RATIOS = {
    "speedup_cached_vs_uncached_sdpa": "uncached_sdpa",
    "speedup_cached_vs_uncached_mha": "uncached_mha",
    "speedup_cached_vs_uncached_default": "uncached_default",
}
RATIO_TOLERANCE = 1e-12  # relative; derived and stored values come from identical arithmetic


def positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def kv_total_ms_median(mode: dict) -> float:
    """Median total generation time of a mode: from raw samples if present, else the stored summary."""
    if "raw" in mode:
        return summarize([1000 * sum(steps) for steps in mode["raw"]["step_seconds"]])["median"]
    return mode["total_ms"]["median"]


def kv_speedups(row: dict) -> dict[str, float]:
    """Derived cache speedups for one KV scenario; raises ValueError on invalid timings."""
    cached = kv_total_ms_median(row["cached"])
    if not positive_finite(cached):
        raise ValueError(f"cached median total time {cached!r} is not a positive finite number")
    ratios = {}
    for name, mode in KV_RATIOS.items():
        if mode in row:
            uncached = kv_total_ms_median(row[mode])
            if not positive_finite(uncached):
                raise ValueError(f"{mode} median total time {uncached!r} is not a positive finite number")
            ratios[name] = uncached / cached
    return ratios


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


def check_kv(key: str, record: dict, problems: list[str], require_raw: bool) -> None:
    """KV-cache records: raw samples (current) or summaries (documented legacy rule), and derived ratios."""
    settings = record["settings"]
    window = record["results"]["window"]
    scenarios = [(row, row["batch"], settings["new_tokens"]) for row in record["results"]["grid"]]
    scenarios.append((window, window["batch"], window["new_tokens"]))
    for row, batch, new_tokens in scenarios:
        where = f"{key} B={batch} prompt={row.get('prompt_len')}"
        valid = True
        for mode in settings["modes"]:
            entry = row[mode]
            if "raw" not in entry:
                if require_raw:
                    problems.append(f"{key}: no raw per-repeat timing samples (historical records must be superseded)")
                    return
                # Legacy rule: summary-only records are checked for internal consistency only.
                values = [entry["total_ms"][stat] for stat in ("median", "p95", "min", "max", "mean")]
                if not all(positive_finite(v) for v in values):
                    problems.append(f"{where}: {mode} summary contains non-positive or non-finite times")
                    valid = False
                continue
            runs = entry["raw"]["step_seconds"]
            if len(runs) != settings["repeats"] or any(len(steps) != new_tokens for steps in runs):
                problems.append(f"{where}: raw sample count does not match repeats x tokens ({mode})")
                valid = False
                continue
            if not all(positive_finite(step) for steps in runs for step in steps):
                problems.append(f"{where}: {mode} raw samples contain non-positive or non-finite times")
                valid = False
                continue
            recomputed = kv_summary_from_raw(entry["raw"], batch, new_tokens)
            if not all(_close(recomputed[name], entry[name]) for name in recomputed):
                problems.append(f"{key}: {mode} B={batch} summaries differ from raw samples")
        if not valid:
            continue
        derived = kv_speedups(row)
        for name, value in derived.items():
            stored = row.get(name)
            if stored is None:
                continue
            if not positive_finite(stored) or not math.isclose(stored, value, rel_tol=RATIO_TOLERANCE, abs_tol=0.0):
                problems.append(f"{where}: stored {name} {stored!r} differs from derived {value:.6g}")


def check_legacy_kv(key: str, record: dict, problems: list[str]) -> None:
    """Legacy single-prompt KV benchmark: medians, speedup and tokens/s derived from its samples."""
    import statistics

    results = record["results"]
    medians = {}
    for mode in ("uncached", "cached"):
        samples = results[f"{mode}_seconds_samples"]
        if not samples or not all(positive_finite(s) for s in samples):
            problems.append(f"{key}: {mode} samples contain non-positive or non-finite times")
            return
        medians[mode] = statistics.median(samples)
        if not math.isclose(results[f"{mode}_seconds"], medians[mode], rel_tol=RATIO_TOLERANCE):
            problems.append(f"{key}: stored {mode}_seconds differs from the median of its samples")
        if not math.isclose(results[f"{mode}_tokens_per_second"], results["generated_tokens"] / medians[mode], rel_tol=RATIO_TOLERANCE):
            problems.append(f"{key}: stored {mode}_tokens_per_second differs from derived value")
    derived = medians["uncached"] / medians["cached"]
    if not positive_finite(results["speedup"]) or not math.isclose(results["speedup"], derived, rel_tol=RATIO_TOLERANCE):
        problems.append(f"{key}: stored speedup {results['speedup']!r} differs from derived {derived:.6g}")


def check_benchmark(key: str, record: dict, problems: list[str], legacy: bool = False) -> None:
    experiment, settings = record["experiment"], record["settings"]
    for name in ("warmup", "repeats", "seed"):
        if name not in settings:
            problems.append(f"{key}: benchmark settings lack {name}")
    if experiment == "kv_cache_scaling":
        check_kv(key, record, problems, require_raw=not legacy)
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


def check_supersession(store: Store, problems: list[str]) -> None:
    """Each entry: both records exist, same kind and experiment, no cycle, chain ends at a current record."""
    for key, entry in store.superseded.items():
        target = entry.get("superseded_by")
        if key not in store.records or target not in store.records:
            problems.append(f"SUPERSEDED.json: {key} -> {target}: both records must exist")
            continue
        if key == target:
            problems.append(f"SUPERSEDED.json: {key} supersedes itself")
            continue
        old, new = store.records[key], store.records[target]
        if key.split("/")[0] != target.split("/")[0] or old["experiment"] != new["experiment"]:
            problems.append(f"SUPERSEDED.json: {key} and {target} are different kinds of record")
        if old["experiment"] == "training":
            if old["settings"].get("seed") != new["settings"].get("seed") or (
                canonical_backend(old["settings"].get("attention_backend")) != canonical_backend(new["settings"].get("attention_backend"))):
                problems.append(f"SUPERSEDED.json: {target} is not the same seed and backend as {key}")
        elif old["experiment"] in {"attention_backends", "kv_cache_scaling", "training_step"}:
            if (old["environment"]["device"], old["settings"].get("dtype")) != (new["environment"]["device"], new["settings"].get("dtype")):
                problems.append(f"SUPERSEDED.json: {target} is not the same device and dtype as {key}")
        elif old["experiment"] == "evaluation" and NAME.match(Path(key).name)["name"] != NAME.match(Path(target).name)["name"]:
            problems.append(f"SUPERSEDED.json: {target} is not the same evaluation as {key}")
        seen, current = {key}, target
        while current in store.superseded:
            if current in seen:
                problems.append(f"SUPERSEDED.json: supersession cycle through {key}")
                break
            seen.add(current)
            current = store.superseded[current].get("superseded_by")
            if current not in store.records:
                problems.append(f"SUPERSEDED.json: chain from {key} ends at a missing record")
                break


def identity_evidence(store: Store, problems: list[str]) -> dict[tuple[str, str], IdentityEvidence]:
    """Resolve best and final checkpoint identities of every current training record once."""
    evidence = {}
    for key, _ in store.current("training"):
        for kind in ("best", "final"):
            found, issues = resolve_identity(store, key, kind)
            evidence[(key, kind)] = found
            problems.extend(issue for issue in issues if issue not in problems)
    return evidence


def validate_store(store: Store, strict: bool = False, require_artifacts: bool = False) -> list[str]:
    """All problems in a record store.

    strict: every current training checkpoint must be artifact_verified or
    trusted_baseline_verified. require_artifacts: every one must be artifact_verified.
    """
    problems: list[str] = []
    check_supersession(store, problems)
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
        if record["experiment"] in {"attention_backends", "kv_cache_scaling", "training_step"}:
            if key not in store.superseded:
                check_benchmark(key, record, problems)
            elif record["experiment"] == "kv_cache_scaling":
                check_benchmark(key, record, problems, legacy=True)
        if record["experiment"] == "legacy_kv_cache_benchmark":
            check_legacy_kv(key, record, problems)
    select_training(store, problems)
    evidence = identity_evidence(store, problems)
    check_attestations(store, problems)
    for (key, kind), found in sorted(evidence.items()):
        if require_artifacts and found.status != ARTIFACT_VERIFIED:
            problems.append(f"{key}: {kind} checkpoint identity is {found.status}; artifact verification required")
        elif strict and found.status not in VERIFIED:
            problems.append(f"{key}: {kind} checkpoint identity is {found.status}; strict verification needs the artifact or the pinned audited baseline")
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


def protocol_rows(store: Store, protocol: str, evaluation_names: list[str], strict: bool = True) -> list[dict]:
    """Training identities of a protocol joined to named evaluations by checkpoint identity.

    Fails closed: raises RecordError on any validation problem, on an evaluation
    row that does not resolve, or on a training identity missing from an evaluation.
    """
    problems = validate_store(store, strict=strict)
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
