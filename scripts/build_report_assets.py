"""Render paper-style monochrome figures from measured experiment records."""

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/data/science_run.json"
OUTPUT = ROOT / "docs/figures"


def label(value: object) -> str:
    return html.escape(str(value), quote=True)


def architecture() -> str:
    stages = [(70, "Input", "UTF-8 bytes"), (270, "Embedding", "token + position"), (470, "Decoder × 4", "attention + SwiGLU"), (670, "Output", "tied LM head")]
    elements = []
    for x, title, subtitle in stages:
        elements.append(f'<rect x="{x}" y="110" width="150" height="82" fill="white" stroke="#1a1a1a" stroke-width="1.4"/><text x="{x + 75}" y="143" text-anchor="middle" class="stage">{title}</text><text x="{x + 75}" y="167" text-anchor="middle" class="minor">{subtitle}</text>')
        if x < 670:
            elements.append(f'<path d="M{x + 156} 151h42m-8-7 8 7-8 7" fill="none" stroke="#1a1a1a" stroke-width="1.4"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 255" role="img" aria-labelledby="title desc">
<title id="title">MiniLM Lab decoder architecture</title><desc id="desc">UTF-8 input is embedded, processed by four causal decoder blocks, and projected to next-token logits with tied output weights.</desc>
<rect width="900" height="255" fill="white"/><text x="48" y="50" class="heading">Figure 1. Decoder architecture</text><text x="48" y="77" class="minor">The benchmark run uses learned positions, RMSNorm and SwiGLU.</text>
{''.join(elements)}<text x="48" y="228" class="foot">Each decoder block: RMSNorm → causal attention → residual → RMSNorm → SwiGLU → residual.</text>
<style>.heading{{font:600 23px Arial,sans-serif;fill:#111}}.stage{{font:600 15px Arial,sans-serif;fill:#111}}.minor{{font:13px Arial,sans-serif;fill:#555}}.foot{{font:12px Arial,sans-serif;fill:#555}}</style></svg>'''


def results(data: dict) -> str:
    evaluation = data["evaluation"]
    benchmark = data["benchmark"]
    metrics = data["metrics"]
    x0, x1, y0, y1 = 92, 496, 158, 390
    losses = [record["validation_loss"] for record in metrics]
    training = [record["train_loss"] for record in metrics]
    lower = math.floor(min(losses + training) * 2) / 2 - 0.25
    upper = math.ceil(max(losses + training) * 2) / 2 + 0.25
    final_step = max(record["step"] for record in metrics)

    def point(step: int, value: float) -> str:
        return f"{x0 + (step / final_step) * (x1 - x0):.1f},{y1 - ((value - lower) / (upper - lower)) * (y1 - y0):.1f}"

    train_line = " ".join(point(record["step"], record["train_loss"]) for record in metrics)
    validation_line = " ".join(point(record["step"], record["validation_loss"]) for record in metrics)
    ticks = []
    for tick in (lower, (lower + upper) / 2, upper):
        y = y1 - ((tick - lower) / (upper - lower)) * (y1 - y0)
        ticks.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dedede"/><text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick:.1f}</text>')
    max_time = max(benchmark["uncached_seconds"], benchmark["cached_seconds"])
    bars = []
    for y, title, seconds, fill in ((212, "Full-context", benchmark["uncached_seconds"], "#303030"), (300, "KV cache", benchmark["cached_seconds"], "#a8a8a8")):
        width = seconds / max_time * 247
        bars.append(f'<text x="575" y="{y - 10}" class="axis">{title}</text><rect x="575" y="{y}" width="{width:.1f}" height="32" fill="{fill}"/><text x="{575 + width + 10:.1f}" y="{y + 22}" class="value">{seconds * 1000:.0f} ms</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 590" role="img" aria-labelledby="title desc">
<title id="title">Science prose language-model results</title><desc id="desc">Full held-out validation loss is {evaluation['validation_loss']:.3f} nats per token, {evaluation['bits_per_byte']:.3f} bits per byte. Median KV cache speedup is {benchmark['speedup']:.2f} times over {benchmark['repeats']} repeats.</desc>
<rect width="900" height="590" fill="white"/>
<text x="48" y="51" class="heading">Figure 2. Science prose experiment</text>
<text x="48" y="76" class="minor">Darwin + Einstein · held-out paragraphs · {label(benchmark['device'].upper())} · {len(metrics)} validation checkpoints</text>
<line x1="48" y1="99" x2="852" y2="99" stroke="#111" stroke-width="1.2"/>
<text x="48" y="131" class="axis">A · Loss by optimizer step (nats/token)</text>
{''.join(ticks)}
<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#111"/>
<polyline points="{train_line}" fill="none" stroke="#111" stroke-width="2.5"/>
<polyline points="{validation_line}" fill="none" stroke="#777" stroke-width="2.5" stroke-dasharray="8 5"/>
<text x="{x0}" y="415" class="small">0</text><text x="{x1}" y="415" text-anchor="end" class="small">{final_step} steps</text>
<line x1="92" y1="446" x2="117" y2="446" stroke="#111" stroke-width="2.5"/><text x="126" y="450" class="small">Train</text>
<line x1="192" y1="446" x2="217" y2="446" stroke="#777" stroke-width="2.5" stroke-dasharray="8 5"/><text x="226" y="450" class="small">Held-out</text>
<text x="555" y="131" class="axis">B · Decode latency, {benchmark['generated_tokens']} tokens</text>
{''.join(bars)}<text x="575" y="392" class="small">Median of {benchmark['repeats']} timed runs; one warm-up per mode</text>
<line x1="48" y1="478" x2="852" y2="478" stroke="#bbb"/>
<text x="48" y="512" class="value">{evaluation['perplexity']:.2f}</text><text x="145" y="512" class="small">held-out perplexity</text>
<text x="374" y="512" class="value">{evaluation['bits_per_byte']:.2f}</text><text x="449" y="512" class="small">bits / byte</text>
<text x="650" y="512" class="value">{benchmark['speedup']:.2f}×</text><text x="719" y="512" class="small">cache speedup</text>
<text x="48" y="554" class="foot">Source: docs/data/science_run.json · One seed; descriptive results, not a confidence interval.</text>
<style>.heading{{font:600 23px Arial,sans-serif;fill:#111}}.axis{{font:600 14px Arial,sans-serif;fill:#222}}.value{{font:600 17px Arial,sans-serif;fill:#111}}.small{{font:12px Arial,sans-serif;fill:#444}}.minor{{font:13px Arial,sans-serif;fill:#555}}.foot{{font:12px Arial,sans-serif;fill:#666}}</style></svg>'''


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "architecture.svg").write_text(architecture(), encoding="utf-8")
    (OUTPUT / "science-results.svg").write_text(results(data), encoding="utf-8")
    print("wrote research figures to docs/figures")


if __name__ == "__main__":
    main()
