"""Headline numbers in the README must be recomputable from experiment records."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
spec = importlib.util.spec_from_file_location("report", ROOT / "scripts/build_experiment_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
RECORDS = report.RECORDS


def attention_ratio_range(keys: list[str]) -> tuple[float, float]:
    records = report.latest(RECORDS / "attention")
    ratios = []
    for key in keys:
        rows = report.rows_by(records[key])
        ratios += [rows[("reference", b, 2048)]["forward_ms"]["median"] / rows[("sdpa", b, 2048)]["forward_ms"]["median"] for b in (1, 4, 8)]
    return min(ratios), max(ratios)


def evaluation(name: str, backend: str, split: str) -> list[float]:
    rows = [r for r in report.latest_by_name(RECORDS / "evaluation")[name]["results"] if report.canon(r["attention_backend"]) == backend]
    return [r["splits"][split]["perplexity"] for r in sorted(rows, key=lambda r: r["seed"])]


def mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def test_attention_speedups_and_memory() -> None:
    low, high = attention_ratio_range(["cpu/float32"])
    assert f"{low:.1f}–{high:.1f}× (CPU)" in README
    low, high = attention_ratio_range(["mps/float32", "mps/float16"])
    assert f"{low:.1f}–{high:.1f}× (MPS)" in README
    memory = []
    for record in report.latest(RECORDS / "attention").values():
        rows = report.rows_by(record)
        memory.append(rows[("reference", 8, 2048)]["memory"]["peak_extra_bytes"] / rows[("sdpa", 8, 2048)]["memory"]["peak_extra_bytes"])
    assert f"{round(min(memory))}–{round(max(memory))}× less extra memory" in README


def test_training_quality_and_step_speed() -> None:
    assert f"{mean_sd(evaluation('sdpa_v1_eval_cpu', 'mha', 'validation'))} (MHA) vs {mean_sd(evaluation('sdpa_v1_eval_cpu', 'sdpa', 'validation'))} (SDPA)" in README
    steps = report.latest(RECORDS / "training_step")

    def gains(key: str) -> list[float]:
        rows = {(report.canon(r["backend"]), r["seq_len"]): r["step_ms"]["median"] for r in steps[key]["results"]}
        return [100 * (rows[("mha", length)] / rows[("sdpa", length)] - 1) for length in (128, 256, 512, 1024)]

    cpu = gains("cpu/float32")
    assert f"{round(min(cpu))}–{round(max(cpu))}% faster with SDPA on CPU" in README
    assert max(abs(g) for g in gains("mps/float32")) < 5 and "within 5% on MPS" in README


def test_kv_cache_claims() -> None:
    records = report.latest(RECORDS / "kv_cache")
    cpu = [row["speedup_cached_vs_uncached_sdpa"] for row in records["cpu/float32"]["results"]["grid"]]
    assert f"{min(cpu):.1f}–{round(max(cpu))}× faster than recomputation on CPU" in README
    mps = [row["speedup_cached_vs_uncached_sdpa"] for row in records["mps/float32"]["results"]["grid"]]
    assert f"({min(mps):.2f}×)" in README
    windows = [r["results"]["window"] for r in records.values()]
    assert all(sum(w["cached"]["refreshed_steps"]) == len(w["cached"]["refreshed_steps"]) - w["context_length"] + w["prompt_len"] for w in windows)


def test_seed_and_v2_numbers() -> None:
    seeds = " / ".join(f"{p:.2f}" for p in evaluation("sdpa_v1_eval_cpu", "mha", "validation"))
    assert f"seeds 42/43/44 give held-out perplexity {seeds}" in README
    assert f"test perplexity is {mean_sd(evaluation('v2_final_cpu', 'mha', 'test'))} (MHA) and {mean_sd(evaluation('v2_final_cpu', 'sdpa', 'test'))} (SDPA)" in README


def test_legacy_v1_values() -> None:
    published = json.loads((ROOT / "docs/data/science_run.json").read_text())
    records = {p.name: json.loads(p.read_text()) for p in sorted((RECORDS / "legacy").glob("*.json"))}
    current = [r for name, r in records.items() if f"legacy/{name}" not in json.loads((RECORDS / "SUPERSEDED.json").read_text())["records"]]
    int8 = next(r["results"] for r in current if r["experiment"] == "legacy_int8_comparison")
    for key in ("original_validation_loss", "int8_validation_loss", "original_stored_bytes", "int8_stored_bytes"):
        assert int8[key] == published["quantization"][key]  # deterministic values reproduce exactly
    kv = next(r["results"] for r in current if r["experiment"] == "legacy_kv_cache_benchmark")
    assert f"{published['benchmark']['speedup']:.2f}×) re-measures at {kv['speedup']:.2f}×" in README
    assert f"{published['evaluation']['perplexity']:.2f}" in README


@pytest.mark.parametrize("path", ["docs/EXPERIMENT_RESULTS.md", "docs/research_report.md"])
def test_detailed_documents_carry_the_same_headline_values(path: str) -> None:
    text = (ROOT / path).read_text()
    assert mean_sd(evaluation("v2_final_cpu", "mha", "test")) in text
    assert "5–28%" in text
