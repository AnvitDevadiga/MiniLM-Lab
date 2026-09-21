"""Small, dependency-free byte and BPE tokenizers for reproducible experiments."""

import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = merges or []
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        sequences = [list(chunk.encode("utf-8")) for chunk in text.splitlines(True)]
        merges = []
        for _ in range(vocab_size - 256):
            counts = Counter(pair for sequence in sequences for pair in pairwise(sequence))
            if not counts:
                break
            pair, _ = counts.most_common(1)[0]
            new_id = 256 + len(merges)
            merges.append(pair)
            updated = []
            for sequence in sequences:
                result = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                updated.append(result)
            sequences = updated
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        # Encode repeated words/punctuation independently and cache them. This
        # avoids repeatedly scanning the entire corpus during preprocessing.
        parts = re.findall(r"\s+|[^\W_]+|[^\w\s]", text, flags=re.UNICODE)
        cache: dict[bytes, tuple[int, ...]] = {}
        encoded = []
        for part in parts:
            raw = part.encode("utf-8")
            if raw not in cache:
                cache[raw] = self._encode_bytes(raw)
            encoded.extend(cache[raw])
        return encoded

    def _encode_bytes(self, raw: bytes) -> tuple[int, ...]:
        tokens = list(raw)
        while len(tokens) > 1:
            candidates = [(self.ranks[(tokens[i], tokens[i + 1])], i) for i in range(len(tokens) - 1) if (tokens[i], tokens[i + 1]) in self.ranks]
            if not candidates:
                break
            _, index = min(candidates)
            pair = (tokens[index], tokens[index + 1])
            tokens[index : index + 2] = [256 + self.ranks[pair]]
        return tuple(tokens)

    def decode(self, tokens: list[int]) -> str:
        def expand(token: int) -> list[int]:
            if token < 256:
                return [token]
            left, right = self.merges[token - 256]
            return expand(left) + expand(right)

        byte_values = [byte for token in tokens for byte in expand(token)]
        return bytes(byte_values).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.merges), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls([tuple(pair) for pair in json.loads(Path(path).read_text(encoding="utf-8"))])
