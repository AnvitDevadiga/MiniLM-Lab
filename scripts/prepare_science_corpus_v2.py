"""Build corpus protocol v2: cleaned prose, train/validation splits and an untouched test book.

v1 (data/science/) is left untouched as a legacy experiment. v2 differs in:
  * non-prose front/back matter removed with exact heading markers per book
  * paragraph rules drop headings, bracketed editorial notes and numeric
    lists/tables (see minilm_lab.corpus); displayed equations are kept
  * a third book by a different author is held out as a test set that is never
    used for training, checkpoint selection or tuning
  * held-out paragraphs that duplicate training text (exact or >= 50% word
    8-gram overlap) are removed and reported

Book bodies are verified by SHA-256 of the extracted Gutenberg body. Raw file
hashes are recorded but may change when Gutenberg edits headers or licences.
Pass --raw-dir to build from previously downloaded pg<ID>.txt files.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from minilm_lab.corpus import (
    clean,
    contamination,
    gutenberg_body,
    main_text,
    paragraphs,
    remove_indices,
    split_by_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "science-v2"
SOURCES = (
    {
        "name": "darwin-1859", "gutenberg_id": 1228, "role": "train_validation",
        "title": "On the Origin of Species (1859)", "author": "Charles Darwin",
        "start": ("ON THE ORIGIN OF SPECIES.", 1), "end": ("INDEX.", 1),
        "body_sha256": "9ff49617f2a0b52b18a2a3b6f09c8a323358160eb98e4e336969b1100768d73f",
    },
    {
        "name": "einstein-1924", "gutenberg_id": 30155, "role": "train_validation",
        "title": "Relativity: The Special and General Theory (Lawson translation)", "author": "Albert Einstein",
        "start": ("PREFACE", 1), "end": ("_K_ = co-ordinate system", 1),
        "body_sha256": "384e707c77e521bec3cceee2b2b6a0e88f271e0f27752a3ec3790ba06554a36e",
    },
    {
        "name": "faraday-1861", "gutenberg_id": 14474, "role": "test",
        "title": "The Chemical History of a Candle", "author": "Michael Faraday",
        "start": ("THE CHEMICAL HISTORY OF A CANDLE", 1), "end": ("NOTES.", 2),
        "body_sha256": "5580d9bfa409ee7f2f7d79b3f27921470a692d6e549a55052adf959547d7ade5",
    },
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(source: dict, raw_dir: Path | None) -> bytes:
    if raw_dir:
        return (raw_dir / f"pg{source['gutenberg_id']}.txt").read_bytes()
    url = f"https://www.gutenberg.org/cache/epub/{source['gutenberg_id']}/pg{source['gutenberg_id']}.txt"
    request = Request(url, headers={"User-Agent": "MiniLM-Lab/1.0 (research corpus preparation)"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def join(parts: list[str]) -> str:
    return "\n\n".join(parts).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/science-v2")
    parser.add_argument("--allow-new-bodies", action="store_true", help="record body hashes instead of verifying them")
    args = parser.parse_args()
    train_parts, validation_parts, test_parts, sources = [], [], [], []
    for source in SOURCES:
        raw = fetch(source, args.raw_dir)
        body = gutenberg_body(raw.decode("utf-8-sig"))
        body_hash = sha256(body.encode("utf-8"))
        if source["body_sha256"] and body_hash != source["body_sha256"] and not args.allow_new_bodies:
            raise ValueError(f"{source['name']} body changed upstream ({body_hash}); inspect before comparing runs")
        text = main_text(body, *source["start"], *source["end"])
        kept, dropped = clean(paragraphs(text))
        entry = {
            "name": source["name"], "title": source["title"], "author": source["author"], "role": source["role"],
            "url": f"https://www.gutenberg.org/cache/epub/{source['gutenberg_id']}/pg{source['gutenberg_id']}.txt",
            "raw_sha256_observed": sha256(raw), "body_sha256": body_hash,
            "start_marker": {"line": source["start"][0], "occurrences": source["start"][1]},
            "end_marker": {"line": source["end"][0], "occurrences": source["end"][1]},
            "main_text_paragraphs": len(kept) + sum(dropped.values()),
            "dropped_paragraphs": dict(sorted(dropped.items())),
            "kept_paragraphs": len(kept),
        }
        if source["role"] == "test":
            test_parts.append(kept)
        else:
            split = split_by_bytes(kept)
            train_parts.append(kept[:split])
            validation_parts.append((source["name"], kept[split:]))
            entry["train_paragraphs"] = split
        sources.append(entry)

    train = [p for part in train_parts for p in part]
    checks = {}
    cleaned_validation = []
    for name, held_out in validation_parts:
        report = contamination(train, held_out)
        cleaned_validation.append((name, remove_indices(held_out, report["exact_duplicates"] + report["near_duplicates"])))
        checks[f"{name}_validation_vs_train"] = report
    test = test_parts[0]
    validation = [p for _, part in cleaned_validation for p in part]
    for label, reference in (("test_vs_train", train), ("test_vs_validation", validation)):
        report = contamination(reference, test)
        checks[label] = report
    test = remove_indices(test, checks["test_vs_train"]["exact_duplicates"] + checks["test_vs_train"]["near_duplicates"]
                          + checks["test_vs_validation"]["exact_duplicates"] + checks["test_vs_validation"]["near_duplicates"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {"train.txt": join(train), "validation.txt": join([join(part) for _, part in cleaned_validation]), "test.txt": join(test)}
    for name, part in cleaned_validation:
        files[f"{name}-validation.txt"] = join(part)
    hashes = {}
    for filename, content in files.items():
        data = content.encode("utf-8")
        (args.output_dir / filename).write_bytes(data)
        hashes[filename] = {"sha256": sha256(data), "bytes": len(data)}
    manifest = {
        "protocol": PROTOCOL,
        "legacy_protocol": "science-v1 in data/science/ (unchanged)",
        "rules": [
            "extract the Project Gutenberg body; verify its SHA-256",
            "keep text from the start marker line (last of the expected occurrences) up to the end marker line",
            "split paragraphs on blank lines",
            "drop paragraphs with no lowercase letter (headings, numerals)",
            "drop paragraphs entirely enclosed in square brackets (illustrations, editorial notes)",
            "drop paragraphs of two or more lines where at least half of the lines end in a number (index, contents, tables)",
            "train/validation books: first ~90% of kept bytes at a paragraph boundary to train, rest to validation",
            "test book: all kept paragraphs; never used for training, checkpoint selection or tuning",
            "remove held-out paragraphs that are exact (normalized) or near (>=50% word 8-gram) duplicates of training or validation text",
        ],
        "usage": {"train": "optimization only", "validation": "checkpoint selection", "test": "final evaluation only, once per selected checkpoint"},
        "sources": sources,
        "files": hashes,
        "contamination_checks": {
            key: {**value, "exact_duplicates": len(value["exact_duplicates"]), "near_duplicates": len(value["near_duplicates"])}
            for key, value in checks.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
