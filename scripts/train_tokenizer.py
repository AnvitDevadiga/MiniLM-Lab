import argparse
from pathlib import Path

from minilm_lab.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bpe.json"))
    args = parser.parse_args()
    tokenizer = BPETokenizer.train(args.text.read_text(encoding="utf-8"), args.vocab_size)
    args.output.parent.mkdir(exist_ok=True)
    tokenizer.save(args.output)
    print(f"saved {args.output} vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
