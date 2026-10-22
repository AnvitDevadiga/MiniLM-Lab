"""Validate every experiment record in experiments/records.

Checks: file naming, required provenance fields, that the recorded commit
exists in this repository, that data hashes in training and evaluation records
match the committed data files, that only superseded records are dirty, and
that docs/EXPERIMENT_RESULTS.md never uses a superseded record.
Exit status is non-zero if any check fails.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"
NAME = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_(?P<name>.+)_(?P<commit>[0-9a-f]{7})\.json$")
ENVIRONMENT_FIELDS = ("python", "torch", "platform", "processor", "device", "torch_threads", "git", "command")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_exists(commit: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], check=False, cwd=ROOT, capture_output=True).returncode == 0


def data_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate() -> tuple[list[str], int]:
    errors = []
    superseded = json.loads((RECORDS / "SUPERSEDED.json").read_text())["records"]
    paths = sorted(RECORDS.glob("*/*.json"))
    checkpoints = set()
    for path in RECORDS.glob("training/*.json"):
        checkpoints.add(json.loads(path.read_text())["results"].get("best_checkpoint_sha256"))
    for relative, entry in superseded.items():
        for path in (relative, entry["superseded_by"]):
            if not (RECORDS / path).exists():
                errors.append(f"SUPERSEDED.json names a missing record: {path}")
    for path in paths:
        relative = str(path.relative_to(RECORDS))
        match = NAME.match(path.name)
        if not match:
            errors.append(f"{relative}: unexpected file name")
            continue
        record = json.loads(path.read_text())
        for key in ("record_version", "created_utc", "experiment", "environment", "settings", "results"):
            if key not in record:
                errors.append(f"{relative}: missing {key}")
        environment = record.get("environment", {})
        missing = [field for field in ENVIRONMENT_FIELDS if field not in environment]
        if missing:
            errors.append(f"{relative}: environment lacks {missing}")
            continue
        git = environment["git"]
        if record["created_utc"] != match["stamp"] or not git["commit"].startswith(match["commit"]):
            errors.append(f"{relative}: file name does not match created_utc/commit")
        if not commit_exists(git["commit"]):
            errors.append(f"{relative}: commit {git['commit']} not in repository")
        if git["dirty"] and relative not in superseded:
            errors.append(f"{relative}: dirty working tree but not listed in SUPERSEDED.json")
        settings = record["settings"]
        if record["experiment"] == "training":
            if not isinstance(settings.get("seed"), int):
                errors.append(f"{relative}: training record without integer seed")
            for split in ("train", "validation"):
                info = settings["data"][split]
                if sha256(data_path(info["path"])) != info["sha256"]:
                    errors.append(f"{relative}: {split} data hash differs from {info['path']}")
            if "best_checkpoint_sha256" not in record["results"]:
                errors.append(f"{relative}: no checkpoint hash")
        elif record["experiment"] == "evaluation":
            for name, info in settings["splits"].items():
                if sha256(data_path(info["path"])) != info["sha256"]:
                    errors.append(f"{relative}: split {name} hash differs from {info['path']}")
            if any(not isinstance(row.get("seed"), int) for row in record["results"]):
                errors.append(f"{relative}: evaluation row without seed")
        elif record["experiment"].startswith("legacy_"):
            if settings.get("checkpoint_sha256") not in checkpoints:
                errors.append(f"{relative}: checkpoint does not match any recorded training run")
            if "validation_text" in settings and sha256(data_path(settings["validation_text"])) != settings["validation_sha256"]:
                errors.append(f"{relative}: validation text hash differs")
        elif record["experiment"] in {"attention_backends", "kv_cache_scaling", "training_step"}:
            for key in ("warmup", "repeats", "seed"):
                if key not in settings:
                    errors.append(f"{relative}: benchmark settings lack {key}")
    # Records selected by the report builder must not be superseded.
    spec = importlib.util.spec_from_file_location("report", ROOT / "scripts/build_experiment_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    used = []
    for directory in ("attention", "attention_quality", "kv_cache", "training_step"):
        used += [r["_path"] for r in report.latest(RECORDS / directory).values()]
    for directory in ("training", "evaluation"):
        used += [r["_path"] for r in report.latest_by_name(RECORDS / directory).values()]
    for path in used:
        relative = str(Path(path).relative_to("experiments/records"))
        if relative in superseded:
            errors.append(f"{relative}: superseded but selected by build_experiment_report.py")
    return errors, len(paths)


def main() -> None:
    errors, count = validate()
    for error in errors:
        print(f"FAIL {error}")
    print(f"validated {count} records: {'OK' if not errors else f'{len(errors)} problems'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
