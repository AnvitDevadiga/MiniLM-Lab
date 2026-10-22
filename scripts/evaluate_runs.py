"""Evaluate selected checkpoints on named splits and write one experiment record.

Each run's best_checkpoint.pt (chosen on validation during training) is scored
with the attention backend it was trained with. On protocol v2 the test split
is evaluated here and nowhere else.

  python scripts/evaluate_runs.py artifacts/runs/v2-*-s4? \
      --split validation=data/science-v2/validation.txt --split test=data/science-v2/test.txt \
      --name v2_final
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.provenance import environment, file_sha256, synchronize, write_record
from minilm_lab.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--split", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--protocol", required=True, help="corpus protocol, e.g. science-v1 or science-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/evaluation")
    args = parser.parse_args()
    splits = {}
    for item in args.split:
        name, path = item.split("=", 1)
        splits[name] = Path(path)
    tokenizer = ByteTokenizer()
    tokens = {name: torch.tensor(tokenizer.encode(path.read_text(encoding="utf-8")), dtype=torch.long) for name, path in splits.items()}
    results = []
    for run in args.runs:
        path = run / "best_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("tokenizer", "byte") != "byte":
            parser.error("only byte-tokenizer checkpoints are supported")
        backend = checkpoint.get("attention_backend", "auto")
        model = MiniLM(MiniLMConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(args.device).set_attention_backend(backend)
        summary_path = run / "run_summary.json"
        row = {
            "run": str(run),
            "seed": checkpoint["seed"],
            "attention_backend": backend,
            "selected_step": checkpoint["step"],
            "selected_on": "validation loss (during training)",
            "checkpoint_sha256": file_sha256(path),
            "corpus_sha256": checkpoint.get("corpus_sha256"),
            "training": json.loads(summary_path.read_text()) if summary_path.exists() else None,
            "splits": {},
        }
        for name, ids in tokens.items():
            synchronize(args.device)
            start = time.perf_counter()
            evaluation = evaluate_tokens(model, ids, args.device)
            synchronize(args.device)
            seconds = time.perf_counter() - start
            row["splits"][name] = {**evaluation, "bits_per_byte": evaluation["validation_loss"] / math.log(2), "eval_seconds": seconds}
            print(f"{run.name} seed={row['seed']} backend={backend} {name}: ppl={evaluation['perplexity']:.3f} ({seconds:.2f}s)", flush=True)
        results.append(row)
    payload = {
        "experiment": "evaluation",
        "environment": environment(args.device),
        "settings": {
            "protocol": args.protocol,
            "splits": {name: {"path": str(path), "sha256": file_sha256(path), "targets": tokens[name].numel() - 1} for name, path in splits.items()},
            "metric": "minilm_lab.metrics.evaluate_tokens (non-overlapping context-length chunks, every target once)",
            "dtype": "float32",
        },
        "results": results,
    }
    print(f"wrote {write_record(args.output_dir, args.name, payload)}")


if __name__ == "__main__":
    main()
