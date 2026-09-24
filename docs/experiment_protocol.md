# Experiment protocol

Use this protocol before publishing a number in the README or research report.

## Fixed controls

- corpus: `data/tinyshakespeare.txt`
- split: contiguous 90% train / 10% validation
- seed: `42`
- optimizer: AdamW, learning rate `3e-4`
- batch size: `8`
- context length: `128`
- evaluation: full validation split where possible, not one random batch
- reporting: validation loss, token perplexity, bits per token, bits per byte, wall time, peak memory, and tokens/sec

## Comparison rules

1. Change one variable at a time.
2. Keep seed, token budget, model width/depth, and evaluation split fixed.
3. Report software versions, backend, device, and commit.
4. Report mean and standard deviation across at least three seeds for claims stronger than a smoke test.
5. Do not rank byte and BPE models by token perplexity alone. Token units differ.
6. For cache tests, assert numerical equivalence before measuring latency.
7. Warm up the device before timing and synchronize asynchronous backends before reading the clock.

## Experiment matrix

| ID | Variable | Primary metric | Secondary metric |
| --- | --- | --- | --- |
| 01 | learned positions vs RoPE | validation loss | length generalization |
| 02 | framework MHA vs SDPA | tokens/sec | peak memory |
| 03 | no cache vs KV cache | decode latency | speedup vs context |
| 04 | model size | loss per training token | memory and throughput |
| 05 | full tuning vs LoRA | quality at fixed budget | trainable parameters |
| 06 | FP32 vs INT8 CPU | latency and size | validation loss |

## Run record

For every run, save:

```text
config, seed, git commit, corpus hash, device, torch version,
train/validation curves, best checkpoint, evaluation JSON, generated samples
```

The benchmark scripts accept `--output` so results can be stored as machine-readable JSON rather than copied from a terminal.
