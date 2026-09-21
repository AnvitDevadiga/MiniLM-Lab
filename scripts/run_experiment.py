"""Launch a named, isolated training run with reproducible artifact paths."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("text", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    output_dir = Path("artifacts/runs") / args.name
    command = [sys.executable, "scripts/train_tiny.py", str(args.text), "--steps", str(args.steps), "--eval-interval", str(args.eval_interval), "--output-dir", str(output_dir)]
    if args.tokenizer:
        command.extend(["--tokenizer", str(args.tokenizer)])
    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"completed {output_dir}")


if __name__ == "__main__":
    main()
