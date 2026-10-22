"""Build attention-experiment tables and SVG figures from recorded runs only.

Tables are written between <!-- BEGIN GENERATED:name --> and
<!-- END GENERATED:name --> markers in docs/EXPERIMENT_RESULTS.md, so the
hand-written analysis stays put while every number comes from a record.
"""

import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/records"
REPORT = ROOT / "docs/EXPERIMENT_RESULTS.md"
FIGURES = ROOT / "docs/figures"
BG, PANEL, EDGE = "#0b1018", "#151d29", "#2c394b"
INK, MUTED = "#f4f7fb", "#a9b8c9"
# Validated categorical slots 1-3 (dark steps) against the panel surface.
SERIES = ("#3987e5", "#d95926", "#199e70")
BACKEND_LABEL = {"auto": "MHA (default)", "sdpa": "SDPA", "reference": "Reference"}
TARGET_ORDER = ("cpu/float32", "mps/float32", "mps/float16", "mps/bfloat16")


def latest(directory: Path) -> dict[str, dict]:
    """Latest record per device/dtype; earlier records stay on disk untouched."""
    chosen = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path.relative_to(ROOT))
        key = f"{record['environment']['device']}/{record['settings'].get('dtype', 'mixed')}"
        chosen[key] = record
    return dict(sorted(chosen.items(), key=lambda item: TARGET_ORDER.index(item[0]) if item[0] in TARGET_ORDER else 99))


def rows_by(record: dict) -> dict[tuple, dict]:
    return {(row["backend"], row["batch"], row["seq_len"]): row for row in record["results"]}


def mib(row: dict) -> str:
    return f"{row['memory']['peak_extra_bytes'] / 2**20:.1f}" if "memory" in row else "–"


def environment_table(records: list[dict]) -> str:
    lines = ["| Record | Device | Dtype | Processor | torch | Python | Threads | Commit | Dirty |", "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |"]
    for record in records:
        env = record["environment"]
        lines.append(
            f"| `{Path(record['_path']).name}` | {env['device']} | {record['settings'].get('dtype', 'mixed')} | {env['processor']} "
            f"| {env['torch']} | {env['python']} | {env['torch_threads']} | `{(env['git']['commit'] or '')[:7]}` | {env['git']['dirty']} |"
        )
    return "\n".join(lines)


def forward_table(key: str, record: dict) -> str:
    rows = rows_by(record)
    settings = record["settings"]
    lines = [
        f"**{key}** · warmup {settings['warmup']}, {settings['repeats']} repeats · median ms (p95) · peak extra MiB per forward",
        "",
        "| B | L | MHA (default) | SDPA | Reference | Ref / SDPA | MHA MiB | SDPA MiB | Ref MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in settings["batch_sizes"]:
        for length in settings["seq_lens"]:
            cells = [rows.get((backend, batch, length)) for backend in ("auto", "sdpa", "reference")]
            if not all(cells):
                continue
            timing = [f"{c['forward_ms']['median']:.2f} ({c['forward_ms']['p95']:.2f})" for c in cells]
            ratio = cells[2]["forward_ms"]["median"] / cells[1]["forward_ms"]["median"]
            lines.append(f"| {batch} | {length} | {' | '.join(timing)} | {ratio:.2f}× | {' | '.join(mib(c) for c in cells)} |")
    return "\n".join(lines)


def decode_table(key: str, record: dict) -> str:
    rows = rows_by(record)
    settings = record["settings"]
    lines = [
        f"**{key}** · {settings['decode_tokens']} greedy decode steps after an L-token prefill · median (p95)",
        "",
        "| B | L | SDPA prefill ms | Ref prefill ms | SDPA decode ms/step | Ref decode ms/step | SDPA gen tok/s | Ref gen tok/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in settings["batch_sizes"]:
        for length in settings["seq_lens"]:
            s, r = rows.get(("sdpa", batch, length)), rows.get(("reference", batch, length))
            if not (s and r):
                continue
            fmt = lambda stat, digits=2: f"{stat['median']:.{digits}f} ({stat['p95']:.{digits}f})"
            lines.append(
                f"| {batch} | {length} | {fmt(s['prefill_ms'])} | {fmt(r['prefill_ms'])} | {fmt(s['decode_ms_per_step'])} "
                f"| {fmt(r['decode_ms_per_step'])} | {s['generation_tokens_per_second']['median']:.0f} | {r['generation_tokens_per_second']['median']:.0f} |"
            )
    return "\n".join(lines)


def variability_table(records: dict[str, dict]) -> str:
    lines = ["| Target | Phase | Median CV | Worst CV | Worst config |", "| --- | --- | ---: | ---: | --- |"]
    for key, record in records.items():
        for phase in ("forward_ms", "prefill_ms", "decode_ms_per_step"):
            values = [(row[phase]["cv"], f"{row['backend']} B={row['batch']} L={row['seq_len']}") for row in record["results"] if phase in row]
            if values:
                ordered = sorted(values)
                lines.append(f"| {key} | {phase} | {ordered[len(ordered) // 2][0]:.3f} | {ordered[-1][0]:.3f} | {ordered[-1][1]} |")
    return "\n".join(lines)


def quality_table(record: dict) -> str:
    settings = record["settings"]
    lines = [
        (
            f"Oracle: {settings['oracle']}; {settings['validation_targets']:,} held-out targets; "
            f"greedy: {settings['greedy']['prompts']} prompts × {settings['greedy']['generated_tokens']} tokens."
        ),
        "",
        "| Seed | Target | Backend | max abs Δlogit | mean abs Δlogit | top-1 agree | Perplexity | Greedy identical |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in record["results"]:
        lines.append(
            f"| {row['seed']} | {row['device']}/{row['dtype']} | {BACKEND_LABEL[row['backend']]} | {row['max_abs_logit_error']:.2e} "
            f"| {row['mean_abs_logit_error']:.2e} | {100 * row['top1_agreement']:.3f}% | {row['perplexity']:.4f} "
            f"| {row['greedy_identical_sequences']}/{row['greedy_prompts']} |"
        )
    if record["settings"].get("skipped_targets"):
        lines.append(f"\nSkipped (unavailable): {', '.join(record['settings']['skipped_targets'])}")
    return "\n".join(lines)


def seed_table(record: dict) -> str:
    rows = [r for r in record["results"] if r["device"] == "cpu" and r["dtype"] == "float32" and r["backend"] == "auto"]
    if not rows:
        return "_No CPU float32 default-backend rows recorded._"
    values = [r["perplexity"] for r in rows]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    lines = ["| Seed | Held-out perplexity | Bits per byte |", "| ---: | ---: | ---: |"]
    lines += [f"| {r['seed']} | {r['perplexity']:.3f} | {r['bits_per_byte']:.4f} |" for r in rows]
    lines.append(f"| **mean ± sd (n={len(values)})** | **{mean:.3f} ± {std:.3f}** | |")
    return "\n".join(lines)


def svg(title: str, description: str, width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>
<rect width="{width}" height="{height}" rx="22" fill="{BG}"/>{body}
<style>
.eyebrow{{font:700 12px Arial,sans-serif;letter-spacing:2px;fill:#67e8f9}}
.title{{font:700 26px Arial,sans-serif;fill:{INK}}}
.subtitle{{font:15px Arial,sans-serif;fill:{MUTED}}}
.label{{font:700 15px Arial,sans-serif;fill:{INK}}}
.axis{{font:12px Arial,sans-serif;fill:{MUTED}}}
.small{{font:12px Arial,sans-serif;fill:{MUTED}}}
</style></svg>"""


def log_panel(x0: float, y0: float, w: float, h: float, heading: str, series: list[tuple], y_label: str, y_fmt, reference_line: float | None = None) -> str:
    """Line panel with log2 x (sequence length) and log10 y. series: (label, color, [(x, y, tip)])."""
    points = [p for _, _, pts in series for p in pts]
    if not points:
        return ""
    xs = sorted({p[0] for p in points})
    low = min(p[1] for p in points + ([(0, reference_line)] if reference_line else []))
    high = max(p[1] for p in points + ([(0, reference_line)] if reference_line else []))
    y_low, y_high = math.floor(math.log10(low) * 4) / 4, math.ceil(math.log10(high) * 4) / 4
    if y_high - y_low < 0.5:
        y_low, y_high = y_low - 0.25, y_high + 0.25
    left, right, top, bottom = x0 + 58, x0 + w - 14, y0 + 40, y0 + h - 34

    def px(x: float) -> float:
        return left + (math.log2(x) - math.log2(xs[0])) / max(math.log2(xs[-1]) - math.log2(xs[0]), 1) * (right - left)

    def py(y: float) -> float:
        return bottom - (math.log10(y) - y_low) / (y_high - y_low) * (bottom - top)

    body = f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="14" fill="{PANEL}" stroke="{EDGE}"/><text x="{x0 + 16}" y="{y0 + 24}" class="label">{html.escape(heading)}</text>'
    ticks = []
    exponent = math.floor(y_low)
    while exponent <= y_high:
        for mantissa in (1, 2, 5):
            value = mantissa * 10**exponent
            if y_low <= math.log10(value) <= y_high:
                ticks.append(value)
        exponent += 1
    for value in ticks:
        y = py(value)
        body += f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{EDGE}" stroke-width="1"/><text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{y_fmt(value)}</text>'
    if reference_line:
        y = py(reference_line)
        body += f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    for x in xs:
        body += f'<text x="{px(x):.1f}" y="{bottom + 18}" text-anchor="middle" class="axis">{x}</text>'
    body += f'<text x="{(left + right) / 2:.1f}" y="{y0 + h - 4}" text-anchor="middle" class="small">sequence length (tokens)</text>'
    body += f'<text transform="translate({x0 + 14},{(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" class="small">{html.escape(y_label)}</text>'
    for _, color, pts in series:
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in pts)
        body += f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        for x, y, tip in pts:
            body += f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" fill="{color}" stroke="{PANEL}" stroke-width="2"><title>{html.escape(tip)}</title></circle>'
    return body


def legend(x: float, y: float, entries: list[tuple[str, str]]) -> str:
    body = ""
    for label, color in entries:
        body += f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="2"/><circle cx="{x + 11}" cy="{y - 4}" r="4.5" fill="{color}"/><text x="{x + 30}" y="{y}" class="axis">{html.escape(label)}</text>'
        x += 40 + 7.2 * len(label)
    return body


def grid_figure(name: str, eyebrow: str, title: str, subtitle: str, footer: str, panels: list[tuple], legend_entries: list[tuple[str, str]]) -> tuple[str, str]:
    columns = 2 if len(panels) > 1 else 1
    width, panel_w, panel_h = 944, (944 - 42 * 2 - 20 * (columns - 1)) / columns, 300
    rows = math.ceil(len(panels) / columns)
    height = 170 + rows * (panel_h + 20) + 40
    body = f'<text x="42" y="46" class="eyebrow">{html.escape(eyebrow)}</text><text x="42" y="84" class="title">{html.escape(title)}</text><text x="42" y="112" class="subtitle">{html.escape(subtitle)}</text>'
    body += legend(42, 146, legend_entries)
    for index, panel in enumerate(panels):
        x = 42 + (index % columns) * (panel_w + 20)
        y = 166 + (index // columns) * (panel_h + 20)
        body += log_panel(x, y, panel_w, panel_h, *panel)
    body += f'<text x="42" y="{height - 18}" class="small">{html.escape(footer)}</text>'
    return name, svg(title, f"{subtitle} {footer}", width, height, body)


def figures(records: dict[str, dict]) -> list[tuple[str, str]]:
    ratio_panels, memory_panels, decode_panels = [], [], []
    for key, record in records.items():
        rows, settings = rows_by(record), record["settings"]
        batches = settings["batch_sizes"]
        ratio_series = []
        for color, batch in zip(SERIES, batches):
            pts = []
            for length in settings["seq_lens"]:
                s, r = rows.get(("sdpa", batch, length)), rows.get(("reference", batch, length))
                if s and r:
                    value = r["forward_ms"]["median"] / s["forward_ms"]["median"]
                    pts.append((length, value, f"{key} B={batch} L={length}: reference/SDPA = {value:.2f}x"))
            ratio_series.append((f"batch {batch}", color, pts))
        ratio_panels.append((key, ratio_series, "reference ÷ SDPA forward time", lambda v: f"{v:g}×", 1.0))
        big = max(batches)
        memory_series = []
        for color, backend in zip(SERIES, ("auto", "sdpa", "reference")):
            pts = [
                (length, rows[(backend, big, length)]["memory"]["peak_extra_bytes"] / 2**20,
                 f"{key} {BACKEND_LABEL[backend]} B={big} L={length}: {rows[(backend, big, length)]['memory']['peak_extra_bytes'] / 2**20:.1f} MiB")
                for length in settings["seq_lens"]
                if (backend, big, length) in rows and "memory" in rows[(backend, big, length)] and rows[(backend, big, length)]["memory"]["peak_extra_bytes"] > 0
            ]
            memory_series.append((BACKEND_LABEL[backend], color, pts))
        memory_panels.append((f"{key} · batch {big}", memory_series, "peak extra memory (MiB)", lambda v: f"{v:g}"))
        decode_series = []
        for color, backend in zip(SERIES[1:], ("sdpa", "reference")):
            pts = [
                (length, rows[(backend, 1, length)]["decode_ms_per_step"]["median"],
                 f"{key} {BACKEND_LABEL[backend]} B=1 L={length}: {rows[(backend, 1, length)]['decode_ms_per_step']['median']:.2f} ms/step")
                for length in settings["seq_lens"] if (backend, 1, length) in rows
            ]
            decode_series.append((BACKEND_LABEL[backend], color, pts))
        decode_panels.append((f"{key} · batch 1", decode_series, "decode ms per token (median)", lambda v: f"{v:g}"))
    backends = [(BACKEND_LABEL[b], c) for c, b in zip(SERIES, ("auto", "sdpa", "reference"))]
    some = next(iter(records.values()))["settings"]
    return [
        grid_figure(
            "attention_speedup.svg", "EXPERIMENT 02 / ATTENTION", "How much faster is fused SDPA than explicit attention?",
            "Full-sequence forward time, reference divided by SDPA (median). Above the dashed line, SDPA is faster.",
            f"Same seeded weights, inputs, dtype and device per panel · {some['repeats']} repeats after {some['warmup']} warmups · source: experiments/records/attention",
            ratio_panels, [(f"batch {b}", c) for c, b in zip(SERIES, some["batch_sizes"])],
        ),
        grid_figure(
            "attention_memory.svg", "EXPERIMENT 02 / MEMORY", "Explicit attention pays for the full score matrix.",
            "Peak extra memory during one forward pass, fresh process per point. Log scales on both axes.",
            "CPU: process peak-RSS delta. MPS: allocator pool growth (approximate high-water mark). Zero deltas are omitted.",
            memory_panels, backends,
        ),
        grid_figure(
            "attention_decode.svg", "EXPERIMENT 02 / DECODE", "Single-token decode against a KV cache.",
            "Median time per generated token at batch 1 as the cached context grows.",
            f"{some['decode_tokens']} greedy steps per trial, each step synchronized and timed · source: experiments/records/attention",
            decode_panels, backends[1:],
        ),
    ]


def replace_section(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN GENERATED:{name} -->\n).*?(\n<!-- END GENERATED:{name} -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"missing generated section markers for {name}")
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def main() -> None:
    speed = latest(RECORDS / "attention")
    quality = latest(RECORDS / "attention_quality")
    if not speed:
        raise SystemExit("no attention records found; run scripts/benchmark_attention.py first")
    sections = {
        "environment": environment_table(list(speed.values()) + list(quality.values())),
        "forward": "\n\n".join(forward_table(key, record) for key, record in speed.items()),
        "decode": "\n\n".join(decode_table(key, record) for key, record in speed.items()),
        "variability": variability_table(speed),
    }
    if quality:
        record = list(quality.values())[-1]
        sections["quality"] = quality_table(record)
        sections["seeds"] = seed_table(record)
    text = REPORT.read_text(encoding="utf-8")
    for name, content in sections.items():
        text = replace_section(text, name, content)
    REPORT.write_text(text, encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, content in figures(speed):
        (FIGURES / name).write_text(content, encoding="utf-8")
    print(f"updated {REPORT.relative_to(ROOT)} and attention figures from {len(speed) + len(quality)} records")


if __name__ == "__main__":
    main()
