#!/usr/bin/env bash
# Release verification. Any failing step (including inside a pipeline) stops the
# script with a non-zero exit status. Run from a clean checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
step() { echo "== $*"; }
step "lint";                "$PY" -m ruff check .
step "tests";               "$PY" -m pytest -q
step "record validation";   "$PY" scripts/validate_records.py | tail -n 1
step "report regeneration"; "$PY" scripts/build_experiment_report.py
step "documentation drift"; git diff --exit-code -- docs/EXPERIMENT_RESULTS.md docs/figures
echo "release verification passed"
