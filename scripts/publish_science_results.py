"""Publish validated local measurements as a compact tracked research record."""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/science-v2"
EVALUATION = ROOT / "artifacts/science-v2-evaluation.json"
BENCHMARK = ROOT / "artifacts/science-v2-kv.json"
INT8 = ROOT / "artifacts/science-v2-int8.json"
DARWIN = ROOT / "artifacts/science-v2-darwin.json"
EINSTEIN = ROOT / "artifacts/science-v2-einstein.json"
MANIFEST = ROOT / "data/science/manifest.json"
OUTPUT = ROOT / "docs/data/science_run.json"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics = read(RUN / "metrics.json")
    evaluation = read(EVALUATION)
    benchmark = read(BENCHMARK)
    int8 = read(INT8)
    per_source = {"darwin-1859": read(DARWIN), "einstein-1924": read(EINSTEIN)}
    manifest = read(MANIFEST)
    checkpoint = torch.load(RUN / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    if not metrics or metrics[-1]["step"] != 1000:
        raise ValueError("expected the completed 1,000-step science run")
    if evaluation["validation_tokens"] < 100000 or benchmark["repeats"] < 3:
        raise ValueError("incomplete validation or benchmark")
    data = {"name": "Public-domain science prose", "seed": checkpoint["seed"], "torch_version": torch.__version__, "config": checkpoint["config"], "corpus": manifest, "metrics": metrics, "evaluation": evaluation, "per_source_evaluation": per_source, "benchmark": benchmark, "quantization": int8}
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"published {OUTPUT}")


if __name__ == "__main__":
    main()
