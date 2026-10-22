"""Validate every experiment record (rules: docs/RECORD_SCHEMA.md).

Checks naming and provenance fields, recorded commits, dataset identities against
the protocol manifests, training duplicates and conflicts, training-to-evaluation
provenance, model-identity evidence, benchmark summaries and derived ratios against
raw samples, and the supersession index. Exits with status 1 if any check fails.

--strict requires every current training checkpoint's model identity to be verified
from the checkpoint artifact or the pinned audited baseline (release gate).
--require-artifacts requires verification from the checkpoint artifacts themselves.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from minilm_lab.records import identity_evidence, load_store, validate_store

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"


def validate(records_root: Path | None = None, strict: bool = False, require_artifacts: bool = False) -> tuple[list[str], int]:
    store = load_store(ROOT, records_root or RECORDS)
    return validate_store(store, strict=strict, require_artifacts=require_artifacts), len(store.records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()
    store = load_store(ROOT, args.records_dir)
    errors = validate_store(store, strict=args.strict, require_artifacts=args.require_artifacts)
    levels = Counter(found.status for found in identity_evidence(store, []).values())
    for error in errors:
        print(f"FAIL {error}")
    print("model identity evidence (checkpoints): " + ", ".join(f"{level}={count}" for level, count in sorted(levels.items())))
    mode = "require-artifacts" if args.require_artifacts else "strict" if args.strict else "standard"
    print(f"validated {len(store.records)} records ({mode}): {'OK' if not errors else f'{len(errors)} problems'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
