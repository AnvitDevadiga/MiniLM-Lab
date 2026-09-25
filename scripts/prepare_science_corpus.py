"""Download two public-domain science books and create document-aware splits."""

import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/science"
SOURCES = (
    ("darwin-1859", "https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4"),
    ("einstein-1924", "https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "86ed8156239455cbb6ed33e06097707d3ab7c40d46e9aa5ff32d3c3f94ef68fd"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def book_body(text: str) -> str:
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg body markers not found")
    return text[text.find("\n", start) + 1 : end].replace("\r\n", "\n").strip() + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_parts, validation_parts, sources = [], [], []
    for name, url, expected_hash in SOURCES:
        request = Request(url, headers={"User-Agent": "MiniLM-Lab/1.0 (research corpus preparation)"})
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        if sha256(raw) != expected_hash:
            raise ValueError(f"upstream text changed for {name}; inspect before running a comparable experiment")
        text = book_body(raw.decode("utf-8-sig"))
        paragraphs = re.split(r"\n\s*\n", text)
        total_bytes = sum(len(paragraph.encode("utf-8")) for paragraph in paragraphs)
        split, used_bytes = 0, 0
        while split < len(paragraphs) - 1 and used_bytes < total_bytes * 0.9:
            used_bytes += len(paragraphs[split].encode("utf-8"))
            split += 1
        if split < 1 or split >= len(paragraphs):
            raise ValueError(f"insufficient paragraphs in {name}")
        training_part = "\n\n".join(paragraphs[:split]).strip() + "\n"
        validation_part = "\n\n".join(paragraphs[split:]).strip() + "\n"
        train_parts.append(training_part)
        validation_parts.append(validation_part)
        (OUTPUT / f"{name}-validation.txt").write_text(validation_part, encoding="utf-8")
        sources.append({"name": name, "url": url, "raw_sha256": sha256(raw), "paragraphs": len(paragraphs), "train_paragraphs": split, "validation_bytes": len(validation_part.encode("utf-8"))})
    train = "\n\n".join(train_parts).encode("utf-8")
    validation = "\n\n".join(validation_parts).encode("utf-8")
    (OUTPUT / "train.txt").write_bytes(train)
    (OUTPUT / "validation.txt").write_bytes(validation)
    manifest = {"split_method": "first approximately 90% of bytes at a paragraph boundary from each document for training; remainder for validation", "sources": sources, "train_bytes": len(train), "validation_bytes": len(validation), "train_sha256": sha256(train), "validation_sha256": sha256(validation)}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
