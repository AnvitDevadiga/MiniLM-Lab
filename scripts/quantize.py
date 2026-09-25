"""Compare FP32 and portable weight-only INT8 on held-out text."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import MiniLM, MiniLMConfig
from minilm_lab.quantization import quantize_linear_weights, stored_parameter_bytes
from minilm_lab.tokenizer import ByteTokenizer


def latency(model: MiniLM, prompt: torch.Tensor, repeats: int) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for index in range(repeats + 1):
            start = time.perf_counter()
            for _ in range(32):
                model(prompt)
            if index:
                values.append(time.perf_counter() - start)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/int8-comparison.json"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("tokenizer") != "byte":
        parser.error("this reference experiment currently supports byte checkpoints")
    model = MiniLM(MiniLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    compact = quantize_linear_weights(copy.deepcopy(model)).eval()
    tokens = torch.tensor(ByteTokenizer().encode(args.validation_text.read_text(encoding="utf-8")), dtype=torch.long)
    prompt = tokens[: model.config.context_length].unsqueeze(0)
    original = evaluate_tokens(model, tokens, "cpu")
    quantized = evaluate_tokens(compact, tokens, "cpu")
    result = {
        "method": "per-output-channel symmetric INT8 linear weights; dequantize on each forward pass",
        "scope": "MLP nn.Linear layers; attention projections and tied LM head remain FP32",
        "device": "cpu",
        "original_validation_loss": original["validation_loss"],
        "int8_validation_loss": quantized["validation_loss"],
        "loss_delta": quantized["validation_loss"] - original["validation_loss"],
        "original_stored_bytes": stored_parameter_bytes(model),
        "int8_stored_bytes": stored_parameter_bytes(compact),
        "original_32_forward_seconds": latency(model, prompt, 5),
        "int8_32_forward_seconds": latency(compact, prompt, 5),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
