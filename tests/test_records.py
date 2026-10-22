import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_experiment_records_are_valid() -> None:
    errors, count = load("validate_records").validate()
    assert count > 0 and errors == []


def test_generated_report_matches_records() -> None:
    # Rebuilding from records must not change the committed document or figures.
    subprocess.run([sys.executable, str(ROOT / "scripts/build_experiment_report.py")], check=True, capture_output=True, cwd=ROOT)
    diff = subprocess.run(["git", "diff", "--stat", "--", "docs/EXPERIMENT_RESULTS.md", "docs/figures"], check=True, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout == ""
