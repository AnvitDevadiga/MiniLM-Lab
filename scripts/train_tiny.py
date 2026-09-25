"""Run a deliberately small local training smoke test on a plain-text file."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from minilm_lab.data import load_text, make_batch, split_tokens
from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def choose_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--validation-text", type=Path, help="pre-split held-out text")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--position", choices=("learned", "rope"), default="learned")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tokenizer", type=Path, default=None, help="BPE JSON artifact; omit for byte tokenizer")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.steps < 1 or args.eval_interval < 1 or args.gradient_accumulation_steps < 1:
        parser.error("steps, eval interval and accumulation must be positive")
    output = args.output_dir / "tiny_checkpoint.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    tokenizer = BPETokenizer.load(args.tokenizer) if args.tokenizer else ByteTokenizer()
    train_text = load_text(args.text)
    ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    if args.validation_text:
        train_ids = ids
        validation_ids = torch.tensor(tokenizer.encode(load_text(args.validation_text)), dtype=torch.long)
    else:
        train_ids, validation_ids = split_tokens(ids)
    if train_ids.numel() <= args.context_length or validation_ids.numel() <= args.context_length:
        raise ValueError("both splits must exceed context_length")
    config = MiniLMConfig(context_length=args.context_length, vocab_size=tokenizer.vocab_size, embedding_dim=args.embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, positional_encoding=args.position)
    model = MiniLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metrics = []
    best_validation_loss = float("inf")
    start_step = 0
    corpus_sha256 = hashlib.sha256(args.text.read_bytes() + (args.validation_text.read_bytes() if args.validation_text else b"")).hexdigest()
    if args.resume and output.exists():
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        if checkpoint["config"] != config.__dict__ or checkpoint.get("corpus_sha256") != corpus_sha256:
            raise ValueError("resume config or corpus differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = checkpoint.get("metrics", [])
        best_validation_loss = checkpoint.get("best_validation_loss", best_validation_loss)
        start_step = checkpoint.get("step", len(metrics))
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        print(f"resumed {output} at step={start_step}")
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        train_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            inputs, targets = make_batch(train_ids, args.batch_size, config.context_length, device)
            _, loss = model(inputs, targets)
            (loss / args.gradient_accumulation_steps).backward()
            train_loss += loss.item() / args.gradient_accumulation_steps
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % args.eval_interval == 0 or step == start_step + args.steps - 1:
            evaluation = evaluate_tokens(model, validation_ids, device)
            validation_loss = evaluation["validation_loss"]
            record = {"step": step + 1, "train_loss": train_loss, **evaluation, "elapsed_seconds": time.perf_counter() - started}
            metrics.append(record)
            print(f"step={step + 1:04d} train_loss={train_loss:.4f} val_loss={validation_loss:.4f} val_ppl={evaluation['perplexity']:.2f} device={device}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save({"model": model.state_dict(), "config": config.__dict__, "metrics": metrics, "best_validation_loss": best_validation_loss, "step": step + 1, "seed": args.seed, "corpus_sha256": corpus_sha256, "tokenizer": str(args.tokenizer) if args.tokenizer else "byte"}, args.output_dir / "best_checkpoint.pt")
    torch.save({"model": model.state_dict(), "config": config.__dict__, "optimizer": optimizer.state_dict(), "rng_state": torch.get_rng_state(), "metrics": metrics, "best_validation_loss": best_validation_loss, "step": start_step + args.steps, "seed": args.seed, "corpus_sha256": corpus_sha256, "tokenizer": str(args.tokenizer) if args.tokenizer else "byte"}, output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
