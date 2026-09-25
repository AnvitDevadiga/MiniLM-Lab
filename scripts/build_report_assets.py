"""Create accessible, data-backed SVG figures for the repository."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/data/science_run.json"
OUTPUT = ROOT / "docs/figures"
BG, PANEL, EDGE = "#0b1018", "#151d29", "#2c394b"
INK, MUTED = "#f4f7fb", "#a9b8c9"
CYAN, LIME, AMBER = "#67e8f9", "#b8f28b", "#ffd17a"


def svg(title: str, description: str, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 944 {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>
<rect width="944" height="{height}" rx="22" fill="{BG}"/>{body}
<style>
.eyebrow{{font:700 12px Arial,sans-serif;letter-spacing:2px;fill:{CYAN}}}
.title{{font:700 31px Arial,sans-serif;fill:{INK}}}
.subtitle{{font:16px Arial,sans-serif;fill:{MUTED}}}
.label{{font:700 18px Arial,sans-serif;fill:{INK}}}
.body{{font:15px Arial,sans-serif;fill:{MUTED}}}
.small{{font:13px Arial,sans-serif;fill:{MUTED}}}
.value{{font:700 34px Arial,sans-serif;fill:{INK}}}
.axis{{font:14px Arial,sans-serif;fill:{MUTED}}}
</style></svg>"""


def architecture(data: dict) -> str:
    cards = [
        (42, "01", "Read text", "Darwin + Einstein", CYAN),
        (265, "02", "Turn into bytes", f"{data['config']['vocab_size']} possible values", LIME),
        (488, "03", "Predict next byte", f"{data['config']['num_layers']} decoder layers", AMBER),
        (711, "04", "Generate text", "Repeat prediction", CYAN),
    ]
    body = '<text x="42" y="52" class="eyebrow">THE IDEA IN FOUR STEPS</text><text x="42" y="94" class="title">A tiny language model, end to end.</text><text x="42" y="126" class="subtitle">It learns which byte is likely to come next. Then it repeats.</text>'
    for x, number, heading, detail, color in cards:
        body += f'<rect x="{x}" y="168" width="193" height="145" rx="16" fill="{PANEL}" stroke="{EDGE}"/><text x="{x+20}" y="202" fill="{color}" class="eyebrow">{number}</text><text x="{x+20}" y="246" class="label">{heading}</text><text x="{x+20}" y="278" class="body">{detail}</text>'
        if x < 711:
            body += f'<path d="M{x+198} 240h19m-7-7 7 7-7 7" fill="none" stroke="{CYAN}" stroke-width="2"/>'
    body += f'<line x1="42" x2="902" y1="342" y2="342" stroke="{EDGE}"/><text x="42" y="372" class="small">Inside the decoder: causal attention · RMSNorm · SwiGLU · tied output weights</text>'
    return svg("How MiniLM Lab works", "Read science text, convert it to bytes, predict the next byte with four decoder layers, and repeat predictions to generate text.", 398, body)


def learning(data: dict) -> str:
    rows = data["metrics"]
    left, right, top, bottom = 105, 865, 195, 465
    last = rows[-1]["step"]

    def xy(step: int, loss: float) -> tuple[float, float]:
        return left + step / last * (right - left), bottom - (loss - 2.5) * (bottom - top)

    body = '<text x="42" y="51" class="eyebrow">EXPERIMENT 01 / LEARNING</text><text x="42" y="93" class="title">Did it learn from the books?</text><text x="42" y="126" class="subtitle">Lower loss means better next-byte predictions. Green is text held back from training.</text>'
    for tick in (2.5, 3.0, 3.5):
        y = xy(0, tick)[1]
        body += f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{EDGE}"/><text x="{left-18}" y="{y+5:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>'
    for row in rows:
        x = xy(row["step"], 2.5)[0]
        body += f'<text x="{x:.1f}" y="493" text-anchor="middle" class="axis">{row["step"]}</text>'
    for key, color in (("train_loss", CYAN), ("validation_loss", LIME)):
        points = " ".join(f"{xy(row['step'], row[key])[0]:.1f},{xy(row['step'], row[key])[1]:.1f}" for row in rows)
        body += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
        for row in rows:
            x, y = xy(row["step"], row[key])
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="{BG}" stroke-width="2"/>'
    body += f'<text x="105" y="532" class="axis">Optimizer updates →</text><circle cx="660" cy="527" r="6" fill="{CYAN}"/><text x="676" y="532" class="axis">Training batch</text><circle cx="793" cy="527" r="6" fill="{LIME}"/><text x="809" y="532" class="axis">Held-out</text>'
    body += f'<line x1="42" x2="902" y1="556" y2="556" stroke="{EDGE}"/><text x="42" y="583" class="small">Four measured checkpoints · 112,111 held-out byte predictions · one seed · source: docs/data/science_run.json</text>'
    description = "Held-out loss by optimizer update: " + "; ".join(
        f"{row['step']} updates, {row['validation_loss']:.3f} nats per byte" for row in rows
    ) + "."
    return svg("Training and held-out loss", description, 607, body)


def speed(data: dict) -> str:
    bench = data["benchmark"]
    slow, fast = bench["uncached_seconds"] * 1000, bench["cached_seconds"] * 1000
    body = f'<text x="42" y="51" class="eyebrow">EXPERIMENT 02 / GENERATION</text><text x="42" y="93" class="title">Reuse past work. Generate faster.</text><text x="42" y="126" class="subtitle">Time for {bench["generated_tokens"]} generated tokens after a {bench["prompt_tokens"]}-token prompt. Shorter is better.</text>'
    for y, name, value, color in ((204, "Recompute every step", slow, AMBER), (309, "Reuse KV cache", fast, CYAN)):
        width = value / slow * 570
        body += f'<text x="42" y="{y-15}" class="label">{name}</text><rect x="250" y="{y-34}" width="{width:.1f}" height="36" rx="8" fill="{color}"/><text x="{min(263+width, 850):.1f}" y="{y-9}" class="label">{value:.1f} ms</text>'
    body += f'<rect x="42" y="364" width="860" height="96" rx="15" fill="{PANEL}" stroke="{EDGE}"/><text x="66" y="425" class="value">{bench["speedup"]:.2f}×</text><text x="206" y="408" class="label">faster with caching</text><text x="206" y="433" class="body">Same sampling settings; CPU median of {bench["repeats"]} runs each.</text>'
    body += '<text x="42" y="497" class="small">CPU result, not an Apple M4/MPS benchmark. Source: docs/data/science_run.json</text>'
    return svg("KV-cache latency", f"For {bench['generated_tokens']} generated tokens, uncached median was {slow:.1f} milliseconds and cached median was {fast:.1f} milliseconds, a {bench['speedup']:.2f}-times speedup on CPU.", 526, body)


def tradeoff(data: dict) -> str:
    q = data["quantization"]
    old, new = q["original_stored_bytes"] / 1_000_000, q["int8_stored_bytes"] / 1_000_000
    reduction = 100 * (1 - new / old)
    old_ms, new_ms = q["original_32_forward_seconds"] * 1000, q["int8_32_forward_seconds"] * 1000
    body = '<text x="42" y="51" class="eyebrow">EXPERIMENT 03 / COMPRESSION</text><text x="42" y="93" class="title">Smaller is not always faster.</text><text x="42" y="126" class="subtitle">INT8 shrinks stored tensors, but this portable reference pays a runtime cost.</text>'
    body += f'<rect x="42" y="164" width="420" height="249" rx="16" fill="{PANEL}" stroke="{EDGE}"/><text x="65" y="204" class="label">Stored tensor size</text><text x="65" y="244" class="body">FP32</text><rect x="157" y="226" width="220" height="24" rx="6" fill="{AMBER}"/><text x="390" y="244" class="small">{old:.2f} MB</text><text x="65" y="292" class="body">INT8</text><rect x="157" y="274" width="{220*new/old:.1f}" height="24" rx="6" fill="{CYAN}"/><text x="{167+220*new/old:.1f}" y="292" class="small">{new:.2f} MB</text><text x="65" y="369" class="value">−{reduction:.1f}%</text><text x="235" y="369" class="body">stored size</text>'
    body += f'<rect x="482" y="164" width="420" height="249" rx="16" fill="{PANEL}" stroke="{EDGE}"/><text x="505" y="204" class="label">32 forward passes</text><text x="505" y="244" class="body">FP32</text><rect x="595" y="226" width="{200*old_ms/new_ms:.1f}" height="24" rx="6" fill="{LIME}"/><text x="809" y="244" class="small">{old_ms:.1f} ms</text><text x="505" y="292" class="body">INT8</text><rect x="595" y="274" width="200" height="24" rx="6" fill="{AMBER}"/><text x="809" y="292" class="small">{new_ms:.1f} ms</text><text x="505" y="369" class="label">INT8 is slower here</text>'
    body += f'<text x="42" y="454" class="small">Only MLP linear weights are quantized. Validation loss changed by +{q["loss_delta"]:.5f} nats/byte.</text><text x="42" y="480" class="small">This does not demonstrate accelerated INT8 inference. Source: docs/data/science_run.json</text>'
    return svg("INT8 storage and runtime trade-off", f"INT8 reduced stored model tensors from {old:.2f} to {new:.2f} megabytes, but 32 forward passes increased from {old_ms:.1f} to {new_ms:.1f} milliseconds.", 508, body)


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "architecture.svg": architecture(data),
        "learning.svg": learning(data),
        "speed.svg": speed(data),
        "tradeoff.svg": tradeoff(data),
    }
    for name, content in figures.items():
        (OUTPUT / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(figures)} figures")


if __name__ == "__main__":
    main()
