"""Write an append-only experiment record for a finished train_tiny.py run."""

import argparse
import json
from pathlib import Path

import torch

from minilm_lab.identity import SCHEME, checkpoint_identity
from minilm_lab.model import canonical_backend
from minilm_lab.provenance import environment, file_sha256, write_record

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train-text", type=Path, default=ROOT / "data/science/train.txt")
    parser.add_argument("--validation-text", type=Path, default=ROOT / "data/science/validation.txt")
    parser.add_argument("--training-command", required=True, help="the exact command that produced the run")
    parser.add_argument("--code-commit", required=True, help="commit whose training code produced the run")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/science/manifest.json")
    parser.add_argument("--protocol", default="science-v1")
    parser.add_argument("--note", default="")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/training")
    args = parser.parse_args()
    final = torch.load(args.run_dir / "tiny_checkpoint.pt", map_location="cpu", weights_only=True)
    summary = args.run_dir / "run_summary.json"
    best = torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    identities = {}
    for label, checkpoint in (("best", best), ("final", final)):
        identity = checkpoint_identity(checkpoint)
        stored = checkpoint.get("model_identity")
        if stored and stored != identity:
            raise ValueError(f"{label} checkpoint's stored model identity does not match its tensors")
        identities[label] = identity["sha256"]
    payload = {
        "experiment": "training",
        "environment": environment("cpu"),
        "settings": {
            "dtype": "float32",
            "seed": final["seed"],
            "protocol": args.protocol,
            "attention_backend": canonical_backend(final.get("attention_backend")),
            "model_config": final["config"],
            "tokenizer": final["tokenizer"],
            "optimizer": "AdamW lr=3e-4 (scripts/train_tiny.py defaults)",
            "steps": final["step"],
            "training_command": args.training_command,
            "training_code_commit": args.code_commit,
            "note": args.note,
            "data": {
                "train": {"path": str(args.train_text.resolve().relative_to(ROOT)), "sha256": file_sha256(args.train_text)},
                "validation": {"path": str(args.validation_text.resolve().relative_to(ROOT)), "sha256": file_sha256(args.validation_text)},
                "corpus_sha256_in_checkpoint": final["corpus_sha256"],
                "manifest_path": str(args.manifest.resolve().relative_to(ROOT)),
                "manifest_sha256": file_sha256(args.manifest),
            },
        },
        "results": {
            "metrics": final["metrics"],
            "best_step": best["step"],
            "best_validation_loss": best["best_validation_loss"],
            "best_checkpoint_sha256": file_sha256(args.run_dir / "best_checkpoint.pt"),
            "final_checkpoint_sha256": file_sha256(args.run_dir / "tiny_checkpoint.pt"),
            "run_summary": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None,
            "model_identity_scheme": SCHEME,
            "best_model_identity_sha256": identities["best"],
            "final_model_identity_sha256": identities["final"],
        },
    }
    name = f"training_{args.protocol}_{canonical_backend(final.get('attention_backend'))}_seed{final['seed']}"
    path = write_record(args.output_dir, name, payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
