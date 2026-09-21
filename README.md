# MiniLM Lab

**A small language model, built from first principles.**

Train it locally. Inspect every layer. Measure what actually changes.

[![CI](https://github.com/AnvitDevadiga/MiniLM-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AnvitDevadiga/MiniLM-Lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%2B-black?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS-black?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

**Anvit Devadiga**

## What is this?

MiniLM Lab is a tiny GPT-style language model that you can train on a MacBook. It learns to predict the next piece of text. The point is not to compete with ChatGPT—the point is to understand how models like it are built, then test each design choice honestly.

## See the idea in one minute

```text
text → tokenizer → Transformer → next-token prediction → generated text
              ↘ validation metrics + benchmarks ↗
```

| Question | What this project measures |
| --- | --- |
| How does text become numbers? | Byte-level tokenizer vs BPE |
| How does the model remember position? | Learned positions vs RoPE |
| How do we train it efficiently? | Checkpoints, accumulation, LoRA |
| How do we generate faster? | Sampling, KV cache, quantization |

## Demo

> **Demo video coming soon.**  
> This space is reserved for a short training → evaluation → generation walkthrough.

<!-- Add the GitHub video attachment or YouTube link here. -->

## Results

Recorded on Apple Silicon MPS using the Tiny Shakespeare corpus.

![MiniLM Lab recorded results](docs/results.svg)

| Run | Best validation perplexity | Inference |
| --- | ---: | ---: |
| Byte tokenizer | **5.14** | **523 tokens/sec** |
| 512-token BPE | **7.99** | More word-like samples |

Perplexity is tokenization-dependent, so the project also records full-split perplexity and bits per byte. See the [research report](docs/research_report.md) for the details and limitations.

## How it works

```text
English text → Byte / BPE tokenizer → Transformer + attention
                                           ↓
                                  next-token prediction
                                           ↓
                              evaluate → save → generate
```

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make check
```

Train an isolated experiment:

```bash
python scripts/run_experiment.py byte-10000 data/tinyshakespeare.txt \
  --steps 10000 --eval-interval 250
```

Generate text from its best checkpoint:

```bash
python scripts/generate.py "To be or not to be" \
  --checkpoint artifacts/runs/byte-10000/best_checkpoint.pt \
  --tokens 200 --top-k 40 --top-p 0.95
```

## Project map

```text
src/minilm_lab/   model, tokenizers, generation, KV cache, LoRA
scripts/          train, evaluate, benchmark, and compare
tests/            correctness and smoke tests
docs/             research report and experiment log
```

## Learn more

- [Research report](docs/research_report.md)
- [Experiment log](docs/experiment_log.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT. Built by Anvit Devadiga.
