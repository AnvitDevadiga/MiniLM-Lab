import pytest

from minilm_lab.corpus import (
    clean,
    contamination,
    gutenberg_body,
    main_text,
    paragraph_drop_reason,
    paragraphs,
    split_by_bytes,
)

BOOK = """Header
*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***
CONTENTS
CHAPTER I. 3

CHAPTER I.

The first paragraph of prose, which we keep.

[Illustration: Fig. 1.]

Oxygen, . . . 20
Nitrogen, . . 80

The value _t_ = 0 is kept.

INDEX.

Aberrant groups, 429.
*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***
licence text
"""


def test_main_text_cuts_front_and_back_matter() -> None:
    text = main_text(gutenberg_body(BOOK), "CHAPTER I.", 1, "INDEX.", 1)
    assert text.startswith("CHAPTER I.\n") and "Aberrant" not in text and "CONTENTS" not in text


def test_marker_counts_are_enforced() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        main_text(gutenberg_body(BOOK), "CHAPTER I.", 2, "INDEX.", 1)


def test_cleaning_rules_keep_prose_and_equations() -> None:
    kept, dropped = clean(paragraphs(main_text(gutenberg_body(BOOK), "CHAPTER I.", 1, "INDEX.", 1)))
    assert kept == ["The first paragraph of prose, which we keep.", "The value _t_ = 0 is kept."]
    assert dropped == {"no_lowercase_heading": 1, "bracketed_editorial": 1, "numeric_list_or_table": 1}


@pytest.mark.parametrize(
    ("paragraph", "reason"),
    [
        ("CHAPTER XIV.\nRECAPITULATION.", "no_lowercase_heading"),
        ("[The Lecturer then lit the candle.]", "bracketed_editorial"),
        ("Agassiz, 139.\nAmblyopsis, 140.", "numeric_list_or_table"),
        ("It was printed in December, 1916", None),
        ("The flame [pointing to it] is bright.", None),
    ],
)
def test_paragraph_rules(paragraph: str, reason: str | None) -> None:
    assert paragraph_drop_reason(paragraph) == reason


def test_split_by_bytes_matches_ninety_percent_rule() -> None:
    parts = ["x" * 10] * 10
    assert split_by_bytes(parts) == 9


def test_contamination_detects_exact_and_near_duplicates() -> None:
    train = ["The quick brown fox jumps over the lazy dog near the old river bank today."]
    held_out = [
        "the QUICK brown fox, jumps over the lazy dog near the old river bank today",  # exact after normalizing
        "The quick brown fox jumps over the lazy dog near the old river bank tomorrow.",  # near
        "Completely different words about candles and flames burning in the lecture hall.",
    ]
    report = contamination(train, held_out)
    assert report["exact_duplicates"] == [0]
    assert report["near_duplicates"] == [1]
