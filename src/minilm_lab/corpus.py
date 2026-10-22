"""Deterministic corpus cleaning, splitting and contamination checks (corpus protocol v2).

Order of operations for every source book:
  1. extract the Project Gutenberg body (between the START and END markers)
  2. keep only the author's main text between two exact heading lines
  3. split into paragraphs on blank lines
  4. drop non-prose paragraphs by the rules in ``paragraph_drop_reason``
  5. split train/validation at a paragraph boundary (train/validation books only)
Then held-out paragraphs that duplicate training text are removed and reported.
"""

import re
from collections import Counter

START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"
# An index, contents or table row: a short line ending in a number.
NUMBERED_LINE = re.compile(r"^.{0,80}?[\s,.]\d{1,4}\.?$")


def gutenberg_body(text: str) -> str:
    start, end = text.find(START), text.find(END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg body markers not found")
    return text[text.find("\n", start) + 1 : end].replace("\r\n", "\n")


def _line_offsets(text: str, line: str) -> list[int]:
    return [match.start() for match in re.finditer(rf"(?m)^{re.escape(line)}[ \t]*$", text)]


def main_text(body: str, start_line: str, start_count: int, end_line: str | None, end_count: int) -> str:
    """Text from the last of ``start_count`` start lines up to the last of ``end_count`` end lines.

    The expected counts are asserted so that upstream edits fail loudly.
    """
    starts = _line_offsets(body, start_line)
    if len(starts) != start_count:
        raise ValueError(f"expected {start_count} '{start_line}' lines, found {len(starts)}")
    end = len(body)
    if end_line is not None:
        ends = _line_offsets(body, end_line)
        if len(ends) != end_count:
            raise ValueError(f"expected {end_count} '{end_line}' lines, found {len(ends)}")
        end = ends[-1]
    if end <= starts[-1]:
        raise ValueError("end marker precedes start marker")
    return body[starts[-1] : end]


def paragraphs(text: str) -> list[str]:
    return [part.strip("\n") for part in re.split(r"\n\s*\n", text) if part.strip()]


def paragraph_drop_reason(paragraph: str) -> str | None:
    stripped = paragraph.strip()
    if not re.search(r"[a-z]", stripped):
        return "no_lowercase_heading"
    if stripped.startswith("[") and stripped.endswith("]"):
        return "bracketed_editorial"
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    # Multi-line only: single lines ending in a number are often equations or dates.
    if len(lines) >= 2 and sum(bool(NUMBERED_LINE.match(line)) for line in lines) * 2 >= len(lines):
        return "numeric_list_or_table"
    return None


def clean(paragraph_list: list[str]) -> tuple[list[str], Counter]:
    kept, dropped = [], Counter()
    for paragraph in paragraph_list:
        reason = paragraph_drop_reason(paragraph)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(paragraph)
    return kept, dropped


def split_by_bytes(paragraph_list: list[str], train_fraction: float = 0.9) -> int:
    """Index of the first held-out paragraph (same rule as corpus protocol v1)."""
    total = sum(len(p.encode("utf-8")) for p in paragraph_list)
    split, used = 0, 0
    while split < len(paragraph_list) - 1 and used < total * train_fraction:
        used += len(paragraph_list[split].encode("utf-8"))
        split += 1
    return split


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def contamination(reference: list[str], candidates: list[str], n: int = 8, threshold: float = 0.5) -> dict:
    """Exact and near-duplicate overlap of candidate paragraphs with reference text.

    Exact: identical after lowercasing and keeping only letters and digits.
    Near: at least ``threshold`` of a paragraph's word n-grams occur in the reference.
    Paragraphs with fewer than n words can only be exact duplicates.
    """
    exact_reference = {normalize(p) for p in reference}
    reference_ngrams = set().union(*(word_ngrams(p, n) for p in reference)) if reference else set()
    exact, near, overlaps = [], [], []
    for index, paragraph in enumerate(candidates):
        if normalize(paragraph) in exact_reference:
            exact.append(index)
            continue
        grams = word_ngrams(paragraph, n)
        if grams:
            share = len(grams & reference_ngrams) / len(grams)
            overlaps.append(share)
            if share >= threshold:
                near.append(index)
    return {
        "paragraphs": len(candidates),
        "exact_duplicates": exact,
        "near_duplicates": near,
        "ngram_n": n,
        "near_threshold": threshold,
        "mean_ngram_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
    }


def remove_indices(paragraph_list: list[str], indices: list[int]) -> list[str]:
    drop = set(indices)
    return [p for i, p in enumerate(paragraph_list) if i not in drop]
