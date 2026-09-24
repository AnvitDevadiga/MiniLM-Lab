"""Run a deliberately small local training smoke test on a plain-text file."""

import argparse
import json
import math
from pathlib import Path

import torch

from minilm_lab.data import load_text, make_batch, split_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def choose_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tokenizer", type=Path, default=None, help="BPE JSON artifact; omit for byte tokenizer")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    output = args.output_dir / "tiny_checkpoint.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = choose_device()
    tokenizer = BPETokenizer.load(args.tokenizer) if args.tokenizer else ByteTokenizer()
    ids = torch.tensor(tokenizer.encode(load_text(args.text)), dtype=torch.long)
    if ids.numel() <= args.context_length * 2 + 1:
        raise ValueError("training text must be longer than the context length")
    train_ids, validation_ids = split_tokens(ids)
    config = MiniLMConfig(context_length=args.context_length, vocab_size=tokenizer.vocab_size)
    model = MiniLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metrics = []
    best_validation_loss = float("inf")
    start_step = 0
    if args.resume and output.exists():
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = checkpoint.get("metrics", [])
        best_validation_loss = checkpoint.get("best_validation_loss", best_validation_loss)
        start_step = checkpoint.get("step", len(metrics)) + 1
        print(f"resumed {output} at step={start_step}")
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, start_step + args.steps):
        inputs, targets = make_batch(train_ids, 8, config.context_length, device)
        _, loss = model(inputs, targets)
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % args.eval_interval == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_x, val_y = make_batch(validation_ids, 8, config.context_length, device)
                _, validation_loss = model(val_x, val_y)
            model.train()
            # Use Python's wider float for logging; float32 overflows for very
            # large early losses even though the perplexity is well-defined.
            validation_perplexity = math.exp(validation_loss.item())
            record = {"step": step, "train_loss": loss.item(), "validation_loss": validation_loss.item(), "validation_perplexity": validation_perplexity}
            metrics.append(record)
            print(f"step={step:04d} train_loss={loss.item():.4f} val_loss={validation_loss.item():.4f} val_ppl={validation_perplexity:.2f} device={device}")
            if validation_loss.item() < best_validation_loss:
                best_validation_loss = validation_loss.item()
                torch.save({"model": model.state_dict(), "config": config.__dict__, "metrics": metrics, "best_validation_loss": best_validation_loss, "step": step, "seed": args.seed, "tokenizer": str(args.tokenizer) if args.tokenizer else "byte"}, args.output_dir / "best_checkpoint.pt")
    torch.save({"model": model.state_dict(), "config": config.__dict__, "optimizer": optimizer.state_dict(), "metrics": metrics, "best_validation_loss": best_validation_loss, "step": start_step + args.steps - 1, "seed": args.seed, "tokenizer": str(args.tokenizer) if args.tokenizer else "byte"}, output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
