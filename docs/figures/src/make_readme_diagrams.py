"""Generate the README diagrams as monochrome, transparent SVGs (light and dark).

Every component, dimension and dispatch rule is taken from src/minilm_lab
(model.py, kv_cache.py, records.py). Run from the repository root:

    python docs/figures/src/make_readme_diagrams.py
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
THEMES = {"light": "#000000", "dark": "#ffffff"}
STROKE = 1.2


class Figure:
    def __init__(self, width: int, height: int, color: str, title: str, description: str) -> None:
        self.width, self.height, self.color = width, height, color
        self.title, self.description = title, description
        self.parts: list[str] = []

    # -------------------------------------------------------------- primitives
    def text(self, x, y, s, size=13, anchor="middle", weight=400, muted=False, rotate=None) -> None:
        opacity = ' fill-opacity="0.68"' if muted else ""
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'dominant-baseline="central"{opacity}{transform}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, dashed=False, radius=3) -> None:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="none"{dash}/>')

    def box(self, x, y, w, h, title, *details, dashed=False, size=13, weight=500) -> None:
        """Box with a title line and muted detail lines, vertically centred."""
        self.rect(x, y, w, h, dashed)
        lines = [(title, size, weight, False)] + [(d, size - 2, 400, True) for d in details]
        step = 15
        top = y + h / 2 - step * (len(lines) - 1) / 2
        for index, (s, font, line_weight, muted) in enumerate(lines):
            self.text(x + w / 2, top + index * step, s, font, weight=line_weight, muted=muted)

    def path(self, points, arrow=True, dashed=False, dotted=False) -> None:
        d = "M " + " L ".join(f"{px} {py}" for px, py in points)
        dash = ' stroke-dasharray="4 3"' if dashed else ' stroke-dasharray="1.5 3"' if dotted else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none"{dash}{marker}/>')

    def plus(self, cx, cy, r=9) -> None:
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none"/>')
        self.parts.append(f'<path d="M {cx - r + 3} {cy} L {cx + r - 3} {cy} M {cx} {cy - r + 3} L {cx} {cy + r - 3}" fill="none"/>')

    def render(self) -> str:
        c = self.color
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">\n'
            f'<title id="title">{escape(self.title)}</title>\n<desc id="desc">{escape(self.description)}</desc>\n'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="{c}" stroke="none"/></marker></defs>\n'
            f'<g fill="{c}" stroke="{c}" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round" '
            f'font-family="{FONT}">\n'
            + "\n".join(p.replace("<text ", '<text stroke="none" ') for p in self.parts)
            + "\n</g>\n</svg>\n"
        )


# ================================================================== architecture


def architecture(color: str) -> Figure:
    f = Figure(720, 712, color, "MiniLM architecture",
               "Decoder-only Transformer: byte token embedding plus learned position embedding, four pre-norm decoder blocks "
               "(RMSNorm, causal self-attention, residual add, RMSNorm, SwiGLU MLP, residual add), final RMSNorm and an LM head "
               "tied to the token embedding.")
    cx, x0, w = 210, 95, 230
    # output and head
    f.text(cx, 18, "next-byte logits  (B, T, 256)", 12, muted=True)
    f.path([(cx, 52), (cx, 30)])
    f.box(x0, 52, w, 40, "Linear (LM head)", "192 → 256, no bias")
    f.path([(cx, 120), (cx, 92)])
    f.box(x0, 120, w, 30, "RMSNorm")
    # decoder block group
    f.rect(52, 168, 298, 372, dashed=True)
    f.text(64, 184, "Decoder block  × 4", 12, anchor="start", weight=500)
    f.plus(cx, 210)
    f.path([(cx, 201), (cx, 150)])
    f.box(x0, 238, w, 42, "SwiGLU MLP", "192 → 768 → 192")
    f.path([(cx, 238), (cx, 219)])
    f.box(x0, 304, w, 28, "RMSNorm")
    f.path([(cx, 304), (cx, 280)])
    f.path([(cx, 363), (cx, 332)])
    f.path([(cx, 352), (78, 352), (78, 210), (201, 210)])  # residual 2
    f.plus(cx, 372)
    f.box(x0, 400, w, 42, "Causal self-attention", "6 heads, d_head = 32")
    f.path([(cx, 400), (cx, 381)])
    f.box(x0, 466, w, 28, "RMSNorm")
    f.path([(cx, 466), (cx, 442)])
    f.path([(cx, 573), (cx, 494)])
    f.path([(cx, 515), (78, 515), (78, 372), (201, 372)])  # residual 1
    # embeddings
    f.plus(cx, 582)
    f.box(x0, 612, w, 40, "Token embedding", "256 × 192")
    f.path([(cx, 612), (cx, 591)])
    f.box(400, 562, 260, 40, "Learned position embedding", "128 × 192 · added unless RoPE")
    f.path([(400, 582), (219, 582)])
    f.text(cx, 690, "input bytes  (B, T),  T ≤ 128", 12, muted=True)
    f.path([(cx, 680), (cx, 652)])
    # weight tying
    f.path([(x0, 632), (30, 632), (30, 72), (x0, 72)], arrow=False, dashed=True)
    f.text(20, 352, "shared weights (tied)", 11, muted=True, rotate=-90)
    # attention detail
    f.path([(x0 + w, 421), (400, 421)], arrow=False, dotted=True)
    f.rect(400, 312, 300, 226, radius=4)
    f.text(412, 328, "Causal self-attention", 12, anchor="start", weight=500)
    f.path([(550, 352), (550, 340)])
    f.box(420, 352, 260, 26, "out_proj  ·  Linear 192 → 192", size=12, weight=400)
    f.box(420, 394, 260, 26, "causal attention  (selectable backend)", size=12, weight=400)
    f.box(420, 438, 260, 26, "RoPE on Q, K  (positional = rope)", dashed=True, size=12, weight=400)
    f.box(420, 482, 260, 26, "in_proj  ·  Linear 192 → 576 → Q, K, V", size=12, weight=400)
    f.path([(550, 526), (550, 508)])
    f.path([(550, 482), (550, 464)])
    f.path([(550, 438), (550, 420)])
    f.path([(550, 394), (550, 378)])
    # MLP detail
    f.path([(x0 + w, 259), (400, 259)], arrow=False, dotted=True)
    f.rect(400, 214, 300, 90, radius=4)
    f.text(412, 230, "SwiGLU MLP", 12, anchor="start", weight=500)
    f.text(550, 258, "down( SiLU(gate(n)) ⊙ up(n) )", 13)
    f.text(550, 282, "gate, up: 192 → 768   ·   down: 768 → 192", 11, muted=True)
    return f


# ================================================================== attention backends


def attention_backends(color: str) -> Figure:
    f = Figure(760, 356, color, "Attention backends",
               "Three interchangeable attention kernels sharing the same weights, and which kernel actually runs for each "
               "call path and selected backend.")
    columns = [(20, "SDPA", "F.scaled_dot_product_attention", "is_causal, or a bottom-right mask"),
               (270, "nn.MultiheadAttention", "explicit boolean causal mask", "projections inside the module"),
               (520, "Reference", "softmax(QKᵀ / √d) · V", "T × T scores, softmax ≥ float32")]
    for x, title, a, b in columns:
        f.box(x, 8, 220, 62, title, a, b)
    f.rect(20, 84, 720, 30)
    f.text(380, 99, "Shared weights:  in_proj 192 → 576  ·  out_proj 192 → 192  ·  6 heads × 32", 12)
    # dispatch table
    f.text(20, 144, "Kernel that runs, by call path (rows) and selected backend (columns)", 12, anchor="start", weight=500)
    top, row, cols = 160, 30, [20, 250, 410, 580, 740]
    headers = ["call path", "sdpa (default)", "mha (legacy v1)", "reference"]
    rows = [("forward, learned positions", "SDPA", "nn.MultiheadAttention", "Reference"),
            ("forward, RoPE", "SDPA", "SDPA *", "Reference"),
            ("forward_cached (KV cache)", "SDPA", "SDPA *", "Reference")]
    f.parts.append(f'<path d="M {cols[0]} {top} L {cols[-1]} {top} M {cols[0]} {top + row} L {cols[-1]} {top + row} '
                   f'M {cols[0]} {top + row * 4} L {cols[-1]} {top + row * 4}" fill="none"/>')
    for index, header in enumerate(headers):
        anchor_x = cols[index] + 8 if index == 0 else (cols[index] + cols[index + 1]) / 2
        f.text(anchor_x, top + row / 2, header, 12, anchor="start" if index == 0 else "middle", weight=500)
    for r, values in enumerate(rows):
        y = top + row * (r + 1) + row / 2
        for index, value in enumerate(values):
            anchor_x = cols[index] + 8 if index == 0 else (cols[index] + cols[index + 1]) / 2
            f.text(anchor_x, y, value, 12, anchor="start" if index == 0 else "middle", muted=index == 0)
    f.text(20, 304, "* RoPE and the KV cache are implemented only on the explicit Q, K, V path, so mha dispatches to SDPA there.",
           11, anchor="start", muted=True)
    f.text(20, 324, "The backend is chosen at run time (set_attention_backend, --attention-backend). It changes no weights",
           11, anchor="start", muted=True)
    f.text(20, 342, "and is not part of the model identity.", 11, anchor="start", muted=True)
    return f


# ================================================================== KV cache


def kv_cache(color: str) -> Figure:
    f = Figure(760, 348, color, "KV-cache inference",
               "Cached decoding: one prefill pass builds per-layer keys and values, then each step processes one new token "
               "and appends its key and value. Uncached decoding recomputes the full window every step.")
    f.text(20, 16, "Cached  ·  forward_cached", 12, anchor="start", weight=500)
    f.box(20, 32, 190, 58, "Prefill", "prompt of P tokens", "one pass over all P positions")
    f.path([(210, 61), (262, 61)])
    f.box(262, 32, 210, 58, "KV cache, per layer", "K, V: (B, 6, P, 32)")
    f.path([(472, 61), (524, 61)])
    f.box(524, 32, 216, 58, "Decode step", "one new token (position P)", "attends to the cache and itself")
    f.path([(632, 90), (632, 112), (367, 112), (367, 90)])
    f.text(500, 126, "append new k, v (torch.cat): cache grows by one position per token", 11, muted=True)
    f.text(20, 152, "Beyond 128 tokens the cache is rebuilt from the last 128 tokens at every step (sliding window).",
           11, anchor="start", muted=True)
    f.text(20, 196, "Uncached  ·  forward", 12, anchor="start", weight=500)
    f.box(20, 212, 452, 58, "Every step: full forward over the last ≤ 128 tokens", "keys and values recomputed for every position")
    f.path([(472, 241), (524, 241)])
    f.box(524, 212, 216, 58, "Logits at last position", "→ next token")
    f.path([(632, 270), (632, 292), (246, 292), (246, 270)])
    f.text(439, 306, "append token, repeat", 11, muted=True)
    f.text(20, 336, "Cache size: 2 × 4 layers × 192 values per position and sequence (fp32: 6,144 bytes).",
           11, anchor="start", muted=True)
    return f


# ================================================================== reproducibility


def reproducibility(color: str) -> Figure:
    f = Figure(884, 470, color, "Reproducibility and verification",
               "Checkpoint files stay local. The repository holds data manifests, training and evaluation records, "
               "self-asserted identity attestations, and an audited baseline whose hash is pinned in code. Strict validation "
               "accepts a model identity only if it is verified from the checkpoint file or from the pinned baseline.")
    # trust zones
    f.rect(12, 12, 220, 446, dashed=True)
    f.text(24, 30, "LOCAL ONLY  ·  not distributed", 10.5, anchor="start", weight=500)
    f.rect(252, 12, 340, 446, dashed=True)
    f.text(264, 30, "REPOSITORY  ·  versioned, reviewed", 10.5, anchor="start", weight=500)
    f.rect(612, 12, 260, 446, dashed=True)
    f.text(624, 30, "VALIDATION", 10.5, anchor="start", weight=500)
    # local
    f.box(30, 52, 184, 44, "train_tiny.py", "seeded training run")
    f.path([(122, 96), (122, 124)])
    f.box(30, 124, 184, 58, "Checkpoint files", "best_checkpoint.pt", "tiny_checkpoint.pt")
    # repository
    f.box(270, 52, 304, 40, "Corpus files and manifests", "SHA-256 per split")
    f.box(270, 108, 304, 58, "Training record  (record_training_run.py)", "command · seed · data hashes",
          "checkpoint SHA-256 · fingerprint")
    f.box(270, 182, 304, 40, "Evaluation record", "joined by checkpoint SHA-256")
    f.box(270, 238, 304, 40, "Identity attestations", "self-asserted: compared, never trusted")
    f.box(270, 300, 304, 40, "Audited baseline  (build_identity_baseline.py)", "checkpoint SHA-256 → model fingerprint")
    f.box(270, 376, 304, 40, "BASELINE_SHA256 pin", "constant in records.py (reviewed code)")
    f.path([(422, 340), (422, 376)], arrow=False, dashed=True)
    f.text(430, 358, "hash of", 10.5, anchor="start", muted=True)
    # checkpoint files -> training record, -> baseline
    f.path([(214, 150), (270, 150)])
    f.path([(122, 182), (122, 320), (270, 320)])
    f.text(130, 226, "bytes = recorded", 10.5, anchor="start", muted=True)
    f.text(130, 242, "SHA-256, then", 10.5, anchor="start", muted=True)
    f.text(130, 258, "fingerprint from", 10.5, anchor="start", muted=True)
    f.text(130, 274, "tensors", 10.5, anchor="start", muted=True)
    # records bus -> validator
    for y in (74, 137, 202, 258, 320):
        f.path([(574, y), (602, y)], arrow=False)
    f.path([(602, 320), (602, 74), (630, 74)])
    # validation
    f.box(630, 52, 204, 44, "validate_records.py", "--strict")
    f.box(630, 128, 204, 96, "Evidence per checkpoint", "artifact_verified  ·  pass", "trusted_baseline_verified  ·  pass",
          "metadata_consistent_only  ·  fail", "unverified  ·  fail")
    f.path([(732, 96), (732, 128)])
    f.box(630, 256, 204, 44, "Report builder", "fails closed on any problem")
    f.path([(732, 224), (732, 256)])
    f.text(732, 326, "baseline trusted only if", 10.5, muted=True)
    f.text(732, 342, "its hash equals the pin", 10.5, muted=True)
    # checkpoint files -> validator, if present (crosses the trust boundary)
    f.path([(60, 182), (60, 440), (852, 440), (852, 74), (834, 74)], dashed=True)
    f.text(422, 430, "if present: rehash bytes, recompute fingerprint", 10.5, muted=True)
    return f


def main() -> None:
    for name, build in (("architecture", architecture), ("attention-backends", attention_backends),
                        ("kv-cache", kv_cache), ("reproducibility", reproducibility)):
        for theme, color in THEMES.items():
            (OUT / f"{name}-{theme}.svg").write_text(build(color).render(), encoding="utf-8")
    print("wrote 8 SVG files to", OUT)


if __name__ == "__main__":
    main()
