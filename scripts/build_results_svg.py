import json
from pathlib import Path


def main() -> None:
    data = json.loads(Path("artifacts/experiment_summary.json").read_text(encoding="utf-8"))
    values = [item["best_validation_perplexity"] for item in data["experiments"]]
    labels = [item["name"] for item in data["experiments"]]
    width, height = 720, 360
    max_value = max(values) * 1.25
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = 150 + index * 250
        bar_height = 220 * value / max_value
        y = 280 - bar_height
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="120" height="{bar_height:.1f}" fill="currentColor" opacity="0.8"/><text x="{x + 60}" y="{y - 10:.1f}" text-anchor="middle">{value:.2f}</text><text x="{x + 60}" y="315" text-anchor="middle">{label.split()[0]}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>Best validation perplexity by tokenizer</title><text x="24" y="32" font-size="22">MiniLM Lab — tokenizer comparison</text><text x="24" y="58">Lower token-level perplexity is not directly comparable across vocabularies.</text><line x1="100" y1="280" x2="650" y2="280" stroke="currentColor"/>{"".join(bars)}</svg>'
    Path("artifacts/tokenizer_comparison.svg").write_text(svg, encoding="utf-8")
    print("saved artifacts/tokenizer_comparison.svg")


if __name__ == "__main__":
    main()
