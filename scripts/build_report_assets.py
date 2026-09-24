"""Build tracked, deterministic SVG figures from the experiment record."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/data/experiment_summary.json"
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
        height = 166 * value / max_ppl
        y = 270 - height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="118" height="{height:.1f}" rx="8" fill="#65d6c5" opacity=".92"/>'
            f'<text x="{x + 59}" y="{y - 12:.1f}" text-anchor="middle" class="value">{value:.2f}</text>'
            f'<text x="{x + 59}" y="302" text-anchor="middle" class="label">{esc(label)}</text>'
        )
    byte = experiments[0]
    bpe = experiments[1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Lab benchmark snapshot</title>
<desc id="desc">Historical Tiny Shakespeare comparison of byte and BPE tokenizers, with validation perplexity and generation throughput.</desc>
<rect width="920" height="520" fill="#0b1220" rx="18"/>
<text x="48" y="58" fill="#f4f7fb" font-family="Arial,sans-serif" font-size="27" font-weight="700">MINILM LAB / BENCHMARK SNAPSHOT</text>
<text x="48" y="88" fill="#94a5bd" font-family="Arial,sans-serif" font-size="14">{esc(data['corpus'])} · {esc(data['hardware'])} · {data['training_steps']} steps · {esc(data['architecture'])}</text>
<g transform="translate(48 122)">
  <rect width="252" height="82" rx="12" fill="#131f31" stroke="#22334a"/>
  <text x="18" y="28" fill="#94a5bd" font-family="Arial,sans-serif" font-size="12">BYTE GENERATION</text>
  <text x="18" y="62" fill="#65d6c5" font-family="Arial,sans-serif" font-size="28" font-weight="700">{byte['generation_tokens_per_second']:.0f}</text>
  <text x="102" y="62" fill="#94a5bd" font-family="Arial,sans-serif" font-size="13">tokens / sec</text>
</g>
<g transform="translate(318 122)">
  <rect width="252" height="82" rx="12" fill="#131f31" stroke="#22334a"/>
  <text x="18" y="28" fill="#94a5bd" font-family="Arial,sans-serif" font-size="12">BPE FULL-SPLIT PPL</text>
  <text x="18" y="62" fill="#a995ff" font-family="Arial,sans-serif" font-size="28" font-weight="700">{bpe['full_split_perplexity']:.2f}</text>
  <text x="106" y="62" fill="#94a5bd" font-family="Arial,sans-serif" font-size="13">validation</text>
</g>
<g transform="translate(588 122)">
  <rect width="284" height="82" rx="12" fill="#131f31" stroke="#22334a"/>
  <text x="18" y="28" fill="#94a5bd" font-family="Arial,sans-serif" font-size="12">BPE BITS / BYTE</text>
  <text x="18" y="62" fill="#ffb86b" font-family="Arial,sans-serif" font-size="28" font-weight="700">{bpe['bits_per_byte']:.4f}</text>
  <text x="150" y="62" fill="#94a5bd" font-family="Arial,sans-serif" font-size="13">tokenizer-normalized</text>
</g>
<text x="48" y="246" fill="#f4f7fb" font-family="Arial,sans-serif" font-size="17" font-weight="700">Best validation perplexity</text>
<text x="48" y="268" fill="#94a5bd" font-family="Arial,sans-serif" font-size="12">Lower is better · token-level values are not directly comparable across vocabularies</text>
<line x1="90" y1="270" x2="830" y2="270" stroke="#33455e"/>
{''.join(bars)}
<text x="48" y="382" fill="#f4f7fb" font-family="Arial,sans-serif" font-size="17" font-weight="700">How to read this</text>
<text x="48" y="409" fill="#c4cfde" font-family="Arial,sans-serif" font-size="14">Byte modeling wins on raw token perplexity and recorded throughput.</text>
<text x="48" y="433" fill="#c4cfde" font-family="Arial,sans-serif" font-size="14">BPE produces fewer, larger units; compare it with bits/byte and samples.</text>
<text x="48" y="480" fill="#667a96" font-family="Arial,sans-serif" font-size="12">Source: docs/data/experiment_summary.json · 128-token CPU inference benchmark</text>
<style>.value{{fill:#f4f7fb;font:700 18px Arial,sans-serif}}.label{{fill:#c4cfde;font:12px Arial,sans-serif}}</style>
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
            rendered.append(f'<path d="M{x - 24} 142H{x - 10}" stroke="#65d6c5" stroke-width="2"/><path d="m{x - 16} 136 8 6-8 6" fill="none" stroke="#65d6c5" stroke-width="2"/>')
        rendered.append(f'<rect x="{x}" y="92" width="160" height="100" rx="14" fill="#141f32" stroke="#2a3c58"/><text x="{x + 80}" y="132" text-anchor="middle" fill="#f4f7fb" font-family="Arial,sans-serif" font-size="15" font-weight="700">{title}</text><text x="{x + 80}" y="158" text-anchor="middle" fill="#94a5bd" font-family="Arial,sans-serif" font-size="12">{subtitle}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Lab system architecture</title>
<desc id="desc">A compact pipeline from raw text through tokenization, Transformer blocks, and a tied language-model head.</desc>
<rect width="860" height="300" fill="#0b1220" rx="18"/>
<text x="42" y="46" fill="#f4f7fb" font-family="Arial,sans-serif" font-size="25" font-weight="700">FROM TEXT TO NEXT TOKEN</text>
<text x="42" y="70" fill="#94a5bd" font-family="Arial,sans-serif" font-size="13">A compact language-model system designed for inspection and measurement</text>
{''.join(rendered)}
<path d="M735 214v28H135v-28" fill="none" stroke="#a995ff" stroke-width="2" stroke-dasharray="5 6"/>
<text x="435" y="270" text-anchor="middle" fill="#a995ff" font-family="Arial,sans-serif" font-size="13">training: next-token loss · inference: sampling + KV cache</text>
</svg>'''


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "results-dashboard.svg").write_text(dashboard(data), encoding="utf-8")
    (FIGURES / "architecture.svg").write_text(architecture(), encoding="utf-8")
    print(f"wrote {FIGURES / 'results-dashboard.svg'}")
    print(f"wrote {FIGURES / 'architecture.svg'}")


if __name__ == "__main__":
    main()
