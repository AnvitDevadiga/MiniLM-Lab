# MiniLM Lab

An English-language, decoder-only Transformer built from first principles in PyTorch.

[![CI](https://github.com/AnvitDevadiga/MiniLM-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AnvitDevadiga/MiniLM-Lab/actions/workflows/ci.yml)

**Author:** Anvit Devadiga

This project is designed for three goals: understand how modern language models work, measure engineering trade-offs honestly, and produce a reproducible public portfolio project.

## Current status

Phase 1 establishes a lightweight, local-first foundation. The implementation will grow through small, tested milestones rather than beginning with an expensive training run.

## Planned components

- byte-level tokenizer and dataset pipeline
- causal self-attention, positional encoding, normalization, and Transformer blocks
- training, evaluation, checkpoints, and perplexity
- autoregressive generation with temperature, top-k, and top-p sampling
- KV-cache benchmarks, LoRA fine-tuning, and quantization experiments
- reproducible ablations with tracked loss, perplexity, throughput, and memory

## Local-first policy

The default configuration is deliberately small and CPU/MPS friendly. No external API or cloud service is required. API-based comparisons, if added later, will be optional and clearly separated from the core implementation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

See `docs/roadmap.md` for the build plan and `configs/tiny.yaml` for the initial safe configuration.

See [the research report](docs/research_report.md) for the current methodology, results, and limitations.

## Verification

```bash
make check
```

The repository intentionally excludes generated checkpoints and run directories. Recreate them with the commands in the research report and experiment log.

## First training run

```bash
source .venv/bin/activate
python scripts/train_tiny.py data/sample.txt --steps 100
```

The script automatically uses Apple Metal (MPS) when available and otherwise falls back to CPU. It keeps a validation split and records validation loss in the checkpoint. The sample corpus is intentionally tiny and is only a smoke test for the pipeline, not a useful language model yet.
