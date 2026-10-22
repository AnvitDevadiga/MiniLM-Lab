"""Numerical and output-quality comparison of attention backends on trained weights.

For each checkpoint, device/dtype and backend:
  * logit error against a float64 CPU oracle (reference backend)
  * next-token top-1 agreement with the oracle
  * held-out loss / perplexity over every validation target (fp32 loss)
  * greedy cached generation agreement with the oracle's tokens
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from minilm_lab.kv_cache import greedy_decode
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.provenance import environment, file_sha256, write_record
from minilm_lab.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(path: Path, device: str, dtype: torch.dtype) -> MiniLM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MiniLM(MiniLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device=device, dtype=dtype).eval()


def windows(tokens: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same chunking as metrics.evaluate_tokens, padded into one batch."""
    count = tokens.numel() - 1
    starts = list(range(1, tokens.numel(), context))
    inputs = torch.zeros(len(starts), context, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    valid = torch.zeros_like(inputs, dtype=torch.bool)
    for row, end in enumerate(starts):
        stop = min(end + context, tokens.numel())
        inputs[row, : stop - end] = tokens[end - 1 : stop - 1]
        targets[row, : stop - end] = tokens[end:stop]
        valid[row, : stop - end] = True
    assert int(valid.sum()) == count
    return inputs, targets, valid


@torch.inference_mode()
def logits_for(model: MiniLM, inputs: torch.Tensor, device: str, batch_size: int = 64) -> torch.Tensor:
    return torch.cat([model(inputs[i : i + batch_size].to(device))[0].cpu().double() for i in range(0, len(inputs), batch_size)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--validation-text", type=Path, default=ROOT / "data/science/validation.txt")
    parser.add_argument("--targets", default="cpu:float32,mps:float32,mps:float16,mps:bfloat16")
    parser.add_argument("--backends", default="mha,sdpa,reference")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/attention_quality")
    args = parser.parse_args()
    targets = [tuple(item.split(":")) for item in args.targets.split(",")]
    skipped = [f"{d}:{t}" for d, t in targets if d == "mps" and not torch.backends.mps.is_available()]
    targets = [(d, t) for d, t in targets if f"{d}:{t}" not in skipped]
    tokens = torch.tensor(ByteTokenizer().encode(args.validation_text.read_text(encoding="utf-8")), dtype=torch.long)
    results = []
    for path in args.checkpoints:
        oracle_model = load(path, "cpu", torch.float64).set_attention_backend("reference")
        context = oracle_model.config.context_length
        inputs, labels, valid = windows(tokens, context)
        oracle = logits_for(oracle_model, inputs, "cpu")
        oracle_top1 = oracle.argmax(-1)
        prompt_starts = torch.linspace(0, tokens.numel() - context - 1, args.prompts).long()
        prompts = torch.stack([tokens[s : s + 32] for s in prompt_starts])
        oracle_generation = greedy_decode(oracle_model, prompts, args.generate_tokens)[:, 32:]
        seed = torch.load(path, map_location="cpu", weights_only=True).get("seed")
        for device, dtype_name in targets:
            model = load(path, device, DTYPES[dtype_name])
            for backend in args.backends.split(","):
                model.set_attention_backend(backend)
                logits = logits_for(model, inputs, device)
                error = (logits - oracle).abs()[valid]
                nll = F.cross_entropy(logits[valid].float(), labels[valid], reduction="sum").item()
                loss = nll / int(valid.sum())
                generated = greedy_decode(model, prompts.to(device), args.generate_tokens)[:, 32:].cpu()
                mismatch = (generated != oracle_generation).int()
                first_divergence = [int(row.argmax()) if row.any() else args.generate_tokens for row in mismatch]
                row = {
                    "checkpoint": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "seed": seed,
                    "device": device,
                    "dtype": dtype_name,
                    "backend": backend,
                    "max_abs_logit_error": error.max().item(),
                    "mean_abs_logit_error": error.mean().item(),
                    "top1_agreement": (logits.argmax(-1) == oracle_top1)[valid].float().mean().item(),
                    "validation_loss": loss,
                    "perplexity": math.exp(loss),
                    "bits_per_byte": loss / math.log(2),
                    "greedy_identical_sequences": int((mismatch.sum(1) == 0).sum()),
                    "greedy_prompts": args.prompts,
                    "greedy_first_divergence": first_divergence,
                }
                results.append(row)
                print(
                    f"seed={seed} {device}/{dtype_name:8s} {backend:9s} max|dlogit|={row['max_abs_logit_error']:.2e} "
                    f"top1={row['top1_agreement']:.5f} ppl={row['perplexity']:.4f} "
                    f"greedy_same={row['greedy_identical_sequences']}/{args.prompts}",
                    flush=True,
                )
    payload = {
        "experiment": "attention_quality",
        "environment": environment(",".join(sorted({d for d, _ in targets}))),
        "settings": {
            "validation_text": str(args.validation_text),
            "validation_sha256": file_sha256(args.validation_text),
            "validation_targets": tokens.numel() - 1,
            "oracle": "cpu float64, reference backend",
            "loss_precision": "logits upcast to float32 before cross-entropy",
            "greedy": {"prompts": args.prompts, "prompt_tokens": 32, "generated_tokens": args.generate_tokens, "cached": True},
            "skipped_targets": skipped,
        },
        "results": results,
    }
    print(f"wrote {write_record(args.output_dir, 'attention_quality', payload)}")


if __name__ == "__main__":
    main()
