"""Build the portfolio-ready technical report from the tracked science run."""

import json
from io import BytesIO
from pathlib import Path

import cairosvg
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/science_run.json"
OUTPUT = ROOT / "output/pdf/MiniLM-Lab-Technical-Report.pdf"
W, H = A4
NAVY = HexColor("#0b1018")
PANEL = HexColor("#151d29")
WHITE = HexColor("#f4f7fb")
INK = HexColor("#17212e")
MUTED = HexColor("#536273")
CYAN = HexColor("#0e8fa5")
PALE = HexColor("#eaf8fa")
EDGE = HexColor("#dce4ec")


def text(c: canvas.Canvas, x: float, y: float, value: str, size: int = 10,
         color=INK, font: str = "Helvetica") -> None:
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, value)


def paragraph(c: canvas.Canvas, value: str, x: float, y: float, width: float,
              size: int = 10, leading: int = 15, color=INK,
              font: str = "Helvetica") -> float:
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if stringWidth(candidate, font, size) > width and line:
            text(c, x, y, line, size, color, font)
            y -= leading
            line = word
        else:
            line = candidate
    if line:
        text(c, x, y, line, size, color, font)
        y -= leading
    return y


def section(c: canvas.Canvas, number: str, title: str, y: float) -> float:
    text(c, 44, y, number, 10, CYAN, "Helvetica-Bold")
    text(c, 73, y, title, 17, INK, "Helvetica-Bold")
    return y - 26


def footer(c: canvas.Canvas, page: int) -> None:
    c.setStrokeColor(EDGE)
    c.line(44, 42, W - 44, 42)
    text(c, 44, 27, "MINILM LAB  /  ANVIT DEVADIGA  /  SEPTEMBER 2026", 8, MUTED, "Helvetica-Bold")
    text(c, W - 75, 27, f"{page} / 4", 8, MUTED)


def figure(c: canvas.Canvas, name: str, x: float, top: float, width: float) -> float:
    image_bytes = cairosvg.svg2png(
        url=str(ROOT / "docs/figures" / f"{name}.svg"), output_width=1888
    )
    image = ImageReader(BytesIO(image_bytes))
    iw, ih = image.getSize()
    height = width * ih / iw
    c.drawImage(image, x, top - height, width=width, height=height)
    return top - height


def cover(c: canvas.Canvas, data: dict) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    text(c, 44, 789, "REPRODUCIBLE ML SYSTEMS STUDY", 10, HexColor("#67e8f9"), "Helvetica-Bold")
    text(c, 44, 735, "MiniLM Lab", 39, WHITE, "Helvetica-Bold")
    text(c, 44, 700, "A small language model, measured end to end", 16, WHITE)
    text(c, 44, 669, "Technical report  /  25 September 2026  /  Anvit Devadiga", 10, HexColor("#a9b8c9"))
    c.linkURL("https://github.com/AnvitDevadiga/MiniLM-Lab", (44, 645, 340, 663), relative=0)
    text(c, 44, 649, "github.com/AnvitDevadiga/MiniLM-Lab", 10, HexColor("#67e8f9"))
    c.setFillColor(PANEL)
    c.roundRect(44, 493, W - 88, 125, 13, fill=1, stroke=0)
    cards = [
        (64, f"{data['evaluation']['perplexity']:.2f}", "held-out perplexity"),
        (237, f"{data['benchmark']['speedup']:.2f}x", "CPU cache speedup"),
        (408, f"{100*(1-data['quantization']['int8_stored_bytes']/data['quantization']['original_stored_bytes']):.1f}%", "stored-size reduction"),
    ]
    for x, value, label in cards:
        text(c, x, 556, value, 27, WHITE, "Helvetica-Bold")
        text(c, x, 526, label, 10, HexColor("#a9b8c9"))
    figure(c, "architecture", 44, 467, W - 88)
    text(c, 44, 234, "Research question", 15, WHITE, "Helvetica-Bold")
    paragraph(c, "Can a compact decoder-only language model learn held-out science prose while revealing measurable inference and storage trade-offs on laptop-class hardware?", 44, 212, W - 88, 11, 17, WHITE)
    paragraph(c, "Scope: one seed, approximately 1 MB of training text, CPU measurements. This is an educational systems study, not a general-purpose chatbot or a production deployment.", 44, 153, W - 88, 10, 15, HexColor("#a9b8c9"))
    text(c, 44, 37, "DATA + CODE + PROTOCOL AVAILABLE IN THE REPOSITORY", 9, HexColor("#67e8f9"), "Helvetica-Bold")
    c.showPage()


def methods(c: canvas.Canvas, data: dict) -> None:
    text(c, 44, 789, "TECHNICAL REPORT", 9, CYAN, "Helvetica-Bold")
    text(c, 44, 752, "What was trained and how?", 25, INK, "Helvetica-Bold")
    y = section(c, "01", "Data and holdout", 708)
    y = paragraph(c, "Public-domain editions of Darwin's On the Origin of Species and Einstein's Relativity were downloaded with pinned SHA-256 hashes. Gutenberg boilerplate was removed. Approximately the first 90% of each book was used for training; the remainder was held out at a paragraph boundary.", 44, y, W - 88)
    y -= 5
    text(c, 44, y, f"{data['corpus']['train_bytes']:,} training bytes", 11, INK, "Helvetica-Bold")
    text(c, 268, y, f"{data['corpus']['validation_bytes']:,} held-out bytes", 11, INK, "Helvetica-Bold")
    y -= 30
    y = section(c, "02", "Model and evaluation", y)
    y = paragraph(c, "Four decoder blocks; width 192; six heads; context 128; byte vocabulary 256; learned positions; RMSNorm; SwiGLU; tied embeddings. AdamW, learning rate 3e-4, microbatch 8, four microbatches per optimizer update, seed 42. The best checkpoint is selected by deterministic full-split validation at updates 250, 500, 750 and 1,000.", 44, y, W - 88)
    y = paragraph(c, "Evaluation scores each next byte once in nonoverlapping 128-token chunks. This fixed-window, within-book holdout is not an unseen-author or knowledge benchmark.", 44, y - 4, W - 88)
    y -= 7
    y = section(c, "03", "Learning curve", y)
    figure_bottom = figure(c, "learning", 44, y, W - 88)
    paragraph(c, f"Final held-out loss: {data['evaluation']['validation_loss']:.4f} nats/byte; perplexity: {data['evaluation']['perplexity']:.2f}; {data['evaluation']['validation_tokens']:,} scored targets. Four checkpoints are shown, not a continuous training trace.", 44, figure_bottom - 14, W - 88, 9, 13, MUTED)
    footer(c, 2)
    c.showPage()


def results(c: canvas.Canvas, data: dict) -> None:
    text(c, 44, 789, "TECHNICAL REPORT", 9, CYAN, "Helvetica-Bold")
    text(c, 44, 752, "What changed in practice?", 25, INK, "Helvetica-Bold")
    y = section(c, "04", "KV-cache latency", 708)
    y = paragraph(c, "For a fixed 21-token prompt and 96 generated tokens, both modes used the same sampling settings. Each had one warm-up and seven timed CPU runs. Reported values are medians; lower latency is better.", 44, y, W - 88)
    y = figure(c, "speed", 44, y - 9, W - 88)
    y -= 29
    y = section(c, "05", "Held-out results by book", y)
    text(c, 44, y, "Source", 10, MUTED, "Helvetica-Bold")
    text(c, 239, y, "Scored bytes", 10, MUTED, "Helvetica-Bold")
    text(c, 358, y, "Perplexity", 10, MUTED, "Helvetica-Bold")
    text(c, 467, y, "Bits/byte", 10, MUTED, "Helvetica-Bold")
    c.setStrokeColor(EDGE)
    c.line(44, y - 9, W - 44, y - 9)
    for offset, key, label in ((33, "darwin-1859", "Darwin"), (61, "einstein-1924", "Einstein")):
        row = data["per_source_evaluation"][key]
        text(c, 44, y - offset, label, 10)
        text(c, 239, y - offset, f"{row['validation_tokens']:,}", 10)
        text(c, 358, y - offset, f"{row['perplexity']:.2f}", 10)
        text(c, 467, y - offset, f"{row['bits_per_byte']:.3f}", 10)
    paragraph(c, "Darwin contributes most of the held-out bytes. These are passages from the same books used for training, not a cross-book transfer test.", 44, y - 91, W - 88, 9, 13, MUTED)
    footer(c, 3)
    c.showPage()


def compression(c: canvas.Canvas, data: dict) -> None:
    text(c, 44, 789, "TECHNICAL REPORT", 9, CYAN, "Helvetica-Bold")
    text(c, 44, 752, "The compression trade-off", 25, INK, "Helvetica-Bold")
    y = section(c, "06", "What INT8 changes", 708)
    y = paragraph(c, "Per-output-channel INT8 quantizes MLP linear weights only. Attention projections, embeddings and the tied output head remain FP32. This portable reference dequantizes weights during each forward pass.", 44, y, W - 88)
    y = figure(c, "tradeoff", 44, y - 15, W - 88)
    y -= 24
    y = section(c, "07", "Limits", y)
    y = paragraph(c, "This study uses one seed, one model size and approximately 1 MB of training text. Results have no confidence interval. The within-book holdout does not show scientific understanding, unseen-author transfer or general chatbot quality.", 44, y, W - 88)
    y = paragraph(c, "CPU timings are local to one execution environment. Apple M4/MPS throughput, peak memory, energy use and accelerated INT8 kernels were not measured. The result supports a storage reduction, not faster INT8 deployment.", 44, y - 6, W - 88)
    y -= 15
    y = section(c, "08", "Reproduce and inspect", y)
    y = paragraph(c, "The repository contains source hashes, prepared-corpus manifest, training and evaluation code, all benchmark samples, tests, and commands. The strongest next experiments are multiple seeds, a genuinely unseen-book holdout and a measured M4 run.", 44, y, W - 88)
    text(c, 44, y - 12, "Code and protocol: github.com/AnvitDevadiga/MiniLM-Lab", 9, CYAN, "Helvetica-Bold")
    c.linkURL("https://github.com/AnvitDevadiga/MiniLM-Lab", (44, y - 16, W - 44, y + 2), relative=0)
    footer(c, 4)
    c.showPage()


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    c.setTitle("MiniLM Lab - Technical Report")
    c.setAuthor("Anvit Devadiga")
    c.setSubject("Reproducible science-prose language-model systems study")
    cover(c, data)
    methods(c, data)
    results(c, data)
    compression(c, data)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
