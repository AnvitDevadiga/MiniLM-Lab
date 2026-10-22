"""Download two public-domain science books and create document-aware splits."""

import argparse
import hashlib
import json
import re
import sys
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


# Published science-v1 split hashes (data/science/manifest.json). Regeneration
# must reproduce these bytes exactly; otherwise nothing is written.
PUBLISHED_TRAIN_SHA256 = "19c057053f222c3084fba3693c1c295ea94f6cda1ee5fbf5f917953fcb91ef00"
PUBLISHED_VALIDATION_SHA256 = "e25d4e45706bbf4251129dbb079f3487660c453cf8b34407901350538b23e2e7"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate and verify the legacy science-v1 corpus.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="use previously downloaded pg<ID>.txt files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    files, sources = {}, []
    train_parts, validation_parts = [], []
    for name, url, expected_hash in SOURCES:
        if args.raw_dir:
            raw = (args.raw_dir / url.rsplit("/", 1)[1]).read_bytes()
        else:
            request = Request(url, headers={"User-Agent": "MiniLM-Lab/1.0 (research corpus preparation)"})
            with urlopen(request, timeout=30) as response:
                raw = response.read()
        if sha256(raw) != expected_hash:
            # Project Gutenberg edits headers and licences; the split hashes below
            # decide whether the published corpus is reproduced.
            print(f"note: raw file for {name} differs from the 2026-09 download ({sha256(raw)})", file=sys.stderr)
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
        files[f"{name}-validation.txt"] = validation_part.encode("utf-8")
        sources.append({"name": name, "url": url, "raw_sha256": sha256(raw), "paragraphs": len(paragraphs), "train_paragraphs": split, "validation_bytes": len(validation_part.encode("utf-8"))})
    files["train.txt"] = "\n\n".join(train_parts).encode("utf-8")
    files["validation.txt"] = "\n\n".join(validation_parts).encode("utf-8")
    if sha256(files["train.txt"]) != PUBLISHED_TRAIN_SHA256 or sha256(files["validation.txt"]) != PUBLISHED_VALIDATION_SHA256:
        raise ValueError("regenerated splits differ from the published science-v1 corpus; nothing was written")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in files.items():
        (args.output_dir / filename).write_bytes(data)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        # The committed manifest is a historical record and is never rewritten.
        manifest = {"split_method": "first approximately 90% of bytes at a paragraph boundary from each document for training; remainder for validation", "sources": sources, "train_bytes": len(files["train.txt"]), "validation_bytes": len(files["validation.txt"]), "train_sha256": sha256(files["train.txt"]), "validation_sha256": sha256(files["validation.txt"])}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"verified science-v1 splits: train {PUBLISHED_TRAIN_SHA256[:12]}, validation {PUBLISHED_VALIDATION_SHA256[:12]}")


if __name__ == "__main__":
    main()
