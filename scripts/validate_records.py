"""Validate every experiment record (rules: docs/RECORD_SCHEMA.md).

Checks naming and provenance fields, recorded commits, dataset identities against
the protocol manifests, training identity conflicts, training-to-evaluation
provenance by checkpoint identity, benchmark summaries against raw samples, and
the supersession index. Exits with status 1 if any check fails.
"""

import argparse
import sys
from pathlib import Path

from minilm_lab.records import load_store, validate_store

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"


def validate(records_root: Path | None = None) -> tuple[list[str], int]:
    store = load_store(ROOT, records_root or RECORDS)
    return validate_store(store), len(store.records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    args = parser.parse_args()
    errors, count = validate(args.records_dir)
    for error in errors:
        print(f"FAIL {error}")
    print(f"validated {count} records: {'OK' if not errors else f'{len(errors)} problems'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
