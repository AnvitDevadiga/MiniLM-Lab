# Experiment protocol

Use this protocol before publishing a number in the README or research report.

## Fixed controls

- corpus: `data/science/train.txt` and `data/science/validation.txt`
- split: approximately 90%/10% by bytes at paragraph boundaries within each source book
- seed: `42`
- optimizer: AdamW, learning rate `3e-4`
- batch size: `8`
- context length: `128`
- evaluation: full validation split at every checkpoint
- reporting: validation loss, token perplexity, bits per byte, measured timing, and stored tensor bytes when relevant

## Comparison rules

1. Change one variable at a time.
2. Keep seed, token budget, model width/depth, and evaluation split fixed.
3. Report software versions, backend, device, source hashes, and commit.
4. Report mean and standard deviation across at least three seeds before claiming a reliable quality difference.
5. Do not rank byte and BPE models by token perplexity alone. Token units differ.
6. For cache tests, assert numerical equivalence before measuring latency.
7. Warm up the device, time repeated runs, use the same sampling settings, and synchronize asynchronous backends before reading the clock.

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
training/validation checkpoints, best checkpoint, evaluation JSON,
and every benchmark timing sample
```

The benchmark scripts accept `--output` so results can be stored as machine-readable JSON rather than copied from a terminal.
