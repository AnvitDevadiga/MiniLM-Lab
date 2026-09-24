"""Build tracked, deterministic SVG figures from the experiment record."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/data/experiment_summary.json"
METRICS_PATH = ROOT / "docs/data/modern_systems_metrics.json"
FIGURES = ROOT / "docs/figures"


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dashboard(data: dict) -> str:
    experiments = data["experiments"]
    labels = [item["name"] for item in experiments]
    perplexities = [item["best_validation_perplexity"] for item in experiments]
    max_ppl = max(perplexities) * 1.2
    bars = []
    for index, (label, value) in enumerate(zip(labels, perplexities)):
        x = 116 + index * 238
        height = 100 * value / max_ppl
        y = 360 - height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="118" height="{height:.1f}" fill="#111111"/>'
            f'<text x="{x + 59}" y="{y - 12:.1f}" text-anchor="middle" class="value">{value:.2f}</text>'
            f'<text x="{x + 59}" y="392" text-anchor="middle" class="label">{esc(label)}</text>'
        )
    byte = experiments[0]
    bpe = experiments[1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="920" height="560" viewBox="0 0 920 560" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Lab benchmark snapshot</title>
<desc id="desc">Historical Tiny Shakespeare comparison of byte and BPE tokenizers, with validation perplexity and generation throughput.</desc>
<rect width="920" height="560" fill="#ffffff"/>
<text x="48" y="58" fill="#111111" font-family="Arial,sans-serif" font-size="27" font-weight="700">MiniLM Lab — benchmark summary</text>
<text x="48" y="88" fill="#555555" font-family="Arial,sans-serif" font-size="14">{esc(data['corpus'])} · {esc(data['hardware'])} · {data['training_steps']} steps · {esc(data['architecture'])}</text>
<g transform="translate(48 122)">
  <rect width="252" height="82" rx="8" fill="#ffffff" stroke="#b8b8b8"/>
  <text x="18" y="28" fill="#555555" font-family="Arial,sans-serif" font-size="12">BYTE GENERATION</text>
  <text x="18" y="62" fill="#111111" font-family="Arial,sans-serif" font-size="28" font-weight="700">{byte['generation_tokens_per_second']:.0f}</text>
  <text x="102" y="62" fill="#555555" font-family="Arial,sans-serif" font-size="13">tokens / sec</text>
</g>
<g transform="translate(318 122)">
  <rect width="252" height="82" rx="8" fill="#ffffff" stroke="#b8b8b8"/>
  <text x="18" y="28" fill="#555555" font-family="Arial,sans-serif" font-size="12">FULL-SPLIT PERPLEXITY</text>
  <text x="18" y="62" fill="#111111" font-family="Arial,sans-serif" font-size="28" font-weight="700">{bpe['full_split_perplexity']:.2f}</text>
  <text x="160" y="62" fill="#555555" font-family="Arial,sans-serif" font-size="13">full split</text>
</g>
<g transform="translate(588 122)">
  <rect width="284" height="82" rx="8" fill="#ffffff" stroke="#b8b8b8"/>
  <text x="18" y="28" fill="#555555" font-family="Arial,sans-serif" font-size="12">BITS PER BYTE</text>
  <text x="18" y="62" fill="#111111" font-family="Arial,sans-serif" font-size="28" font-weight="700">{bpe['bits_per_byte']:.4f}</text>
  <text x="150" y="62" fill="#555555" font-family="Arial,sans-serif" font-size="13">normalized</text>
</g>
<text x="48" y="250" fill="#111111" font-family="Arial,sans-serif" font-size="17" font-weight="700">Best validation perplexity</text>
<text x="48" y="272" fill="#555555" font-family="Arial,sans-serif" font-size="12">Lower is better · token-level values are not directly comparable across vocabularies</text>
<line x1="90" y1="360" x2="830" y2="360" stroke="#111111"/>
{''.join(bars)}
<text x="48" y="438" fill="#111111" font-family="Arial,sans-serif" font-size="17" font-weight="700">Interpretation</text>
<text x="48" y="465" fill="#333333" font-family="Arial,sans-serif" font-size="14">The systems corpus reaches lower best-step perplexity under the same compact setup.</text>
<text x="48" y="489" fill="#333333" font-family="Arial,sans-serif" font-size="14">Compare corpora with full-split perplexity, bits per byte, and matched samples.</text>
<text x="48" y="532" fill="#777777" font-family="Arial,sans-serif" font-size="12">Source: docs/data/experiment_summary.json · 128-token CPU inference benchmark</text>
<style>.value{{fill:#111111;font:700 18px Arial,sans-serif}}.label{{fill:#333333;font:12px Arial,sans-serif}}</style>
</svg>'''


def architecture() -> str:
    nodes = [
        (55, "RAW TEXT", "UTF-8 bytes / BPE"),
        (255, "TOKEN IDs", "256–512 vocabulary"),
        (455, "TRANSFORMER", "RoPE · RMSNorm · SwiGLU"),
        (655, "LM HEAD", "tied embeddings"),
    ]
    rendered = []
    for index, (x, title, subtitle) in enumerate(nodes):
        if index:
            rendered.append(f'<path d="M{x - 24} 142H{x - 10}" stroke="#111111" stroke-width="2"/><path d="m{x - 16} 136 8 6-8 6" fill="none" stroke="#111111" stroke-width="2"/>')
        rendered.append(f'<rect x="{x}" y="92" width="160" height="100" rx="8" fill="#ffffff" stroke="#111111"/><text x="{x + 80}" y="132" text-anchor="middle" fill="#111111" font-family="Arial,sans-serif" font-size="15" font-weight="700">{title}</text><text x="{x + 80}" y="158" text-anchor="middle" fill="#555555" font-family="Arial,sans-serif" font-size="12">{subtitle}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Lab system architecture</title>
<desc id="desc">A compact pipeline from raw text through tokenization, Transformer blocks, and a tied language-model head.</desc>
<rect width="860" height="300" fill="#ffffff"/>
<text x="42" y="46" fill="#111111" font-family="Arial,sans-serif" font-size="25" font-weight="700">From text to next token</text>
<text x="42" y="70" fill="#555555" font-family="Arial,sans-serif" font-size="13">A compact language-model system designed for inspection and measurement</text>
{''.join(rendered)}
<path d="M735 214v28H135v-28" fill="none" stroke="#777777" stroke-width="2" stroke-dasharray="5 6"/>
<text x="435" y="270" text-anchor="middle" fill="#333333" font-family="Arial,sans-serif" font-size="13">training: next-token loss · inference: sampling + KV cache</text>
</svg>'''


def training_curve(metrics: list[dict]) -> str:
    width, height = 920, 520
    left, top, plot_width, plot_height = 84, 96, 780, 280
    max_loss = max(item["validation_loss"] for item in metrics + [{"validation_loss": 0}])
    max_loss = max(max_loss, max(item["train_loss"] for item in metrics))

    def points(key: str) -> str:
        return " ".join(
            f"{left + item['step'] / 1000 * plot_width:.1f},{top + plot_height - item[key] / max_loss * plot_height:.1f}"
            for item in metrics
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Systems corpus training curves</title>
<desc id="desc">Training and validation cross-entropy loss over 1,000 optimization steps.</desc>
<rect width="920" height="520" fill="#ffffff"/>
<text x="48" y="52" fill="#111111" font-family="Arial,sans-serif" font-size="26" font-weight="700">Training dynamics — MiniLM Systems corpus</text>
<text x="48" y="78" fill="#555555" font-family="Arial,sans-serif" font-size="13">RMSNorm + SwiGLU · byte tokenizer · 1,000 steps · CPU validation run</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111"/>
<line x1="{left}" y1="{top + plot_height / 2}" x2="{left + plot_width}" y2="{top + plot_height / 2}" stroke="#dddddd"/>
<polyline points="{points('train_loss')}" fill="none" stroke="#111111" stroke-width="3"/>
<polyline points="{points('validation_loss')}" fill="none" stroke="#777777" stroke-width="3" stroke-dasharray="8 6"/>
<text x="{left}" y="404" fill="#333333" font-family="Arial,sans-serif" font-size="12">0</text>
<text x="{left + plot_width - 24}" y="404" fill="#333333" font-family="Arial,sans-serif" font-size="12">1,000 steps</text>
<text x="20" y="{top + plot_height / 2}" fill="#333333" font-family="Arial,sans-serif" font-size="12" transform="rotate(-90 20 {top + plot_height / 2})">cross-entropy loss</text>
<line x1="{left + 12}" y1="450" x2="{left + 48}" y2="450" stroke="#111111" stroke-width="3"/><text x="{left + 58}" y="455" fill="#333333" font-family="Arial,sans-serif" font-size="13">training loss</text>
<line x1="{left + 190}" y1="450" x2="{left + 226}" y2="450" stroke="#777777" stroke-width="3" stroke-dasharray="8 6"/><text x="{left + 236}" y="455" fill="#333333" font-family="Arial,sans-serif" font-size="13">validation loss</text>
<text x="48" y="492" fill="#777777" font-family="Arial,sans-serif" font-size="12">Source: docs/data/modern_systems_metrics.json</text>
</svg>'''


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "results-dashboard.svg").write_text(dashboard(data), encoding="utf-8")
    (FIGURES / "architecture.svg").write_text(architecture(), encoding="utf-8")
    (FIGURES / "training-curves.svg").write_text(training_curve(metrics), encoding="utf-8")
    print(f"wrote {FIGURES / 'results-dashboard.svg'}")
    print(f"wrote {FIGURES / 'architecture.svg'}")
    print(f"wrote {FIGURES / 'training-curves.svg'}")


if __name__ == "__main__":
    main()
