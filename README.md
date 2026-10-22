# MiniLM Lab

A small decoder-only Transformer, trained on byte-level science prose, used as a laboratory for measuring how implementation choices affect quality, latency, throughput and memory. Every reported number is recomputed from a validated record in [`experiments/records/`](experiments/records/).

## Architecture

![MiniLM Lab architecture](docs/figures/architecture-light.svg#gh-light-mode-only)
![MiniLM Lab architecture](docs/figures/architecture-dark.svg#gh-dark-mode-only)

Defaults: d = 192, 4 layers, 6 heads, context 128, vocabulary 256 (UTF-8 bytes). Options: RoPE instead of learned positions, LayerNorm instead of RMSNorm, GELU MLP instead of SwiGLU.

## Key Results

Apple M4 (16 GB), CPU and MPS, Python 3.12.14, torch 2.14.1. Full tables: [EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md).

| Experiment | Result |
| --- | --- |
| Attention backends, fp32, 3 trained seeds | identical held-out perplexity, top-1 predictions and greedy text across `sdpa`, `mha` and `reference` |
| Reference ÷ SDPA forward time, 2,048 tokens | 3.4–3.9× (CPU), 3.9–5.4× (MPS); SDPA uses 8–22× less extra memory at batch 8 |
| Training, legacy MHA vs SDPA (corpus v1, 3 seeds) | validation perplexity 19.21 ± 1.75 (MHA) vs 19.04 ± 1.37 (SDPA); training step 5–28% faster with SDPA on CPU, within 5% on MPS |
| KV cache vs recomputation, 64 tokens | 2.2–23× faster than recomputation on CPU; slower on MPS for batch-1 prompts of 1–16 tokens (0.88×) |
| Seed variance, published v1 configuration | seeds 42/43/44 give held-out perplexity 17.58 / 18.99 / 21.07; the published 17.58 is the best of three |
| Held-out test book (corpus v2, Faraday) | test perplexity is 15.66 ± 2.58 (MHA) and 15.60 ± 2.63 (SDPA) |

Corpus v1 and v2 use different data and preprocessing; their perplexities are not comparable.

## Implementation

| Component | Implementation |
| --- | --- |
| Model | decoder-only, pre-norm residual blocks, RMSNorm, SwiGLU, learned positions or RoPE, tied embeddings ([`model.py`](src/minilm_lab/model.py)) |
| Attention backends | `sdpa` (default), `mha` (legacy path of all v1 results; alias `auto`), `reference`; same weights, selected at run time ([diagram](docs/figures/attention-backends-light.svg) · [dark](docs/figures/attention-backends-dark.svg)) |
| Inference | KV cache with prefill, single-token decode and sliding-window refresh ([diagram](docs/figures/kv-cache-light.svg) · [dark](docs/figures/kv-cache-dark.svg)); uncached sampling with temperature, top-k and top-p; cached sampling with temperature and greedy decoding |
| Training | AdamW, gradient accumulation, full-split checkpoint selection, bitwise resume with optimizer and RNG state |
| Evaluation | every held-out next byte scored once; perplexity and bits per byte |
| Provenance | append-only records, deterministic model fingerprints, audited baseline, strict validation ([diagram](docs/figures/reproducibility-light.svg) · [dark](docs/figures/reproducibility-dark.svg)) |

## Quick Start

Python 3.12. The lock file pins the exact packages used for every record.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/release-lock.txt
python -m pip install -e . --no-deps
make check                                   # ruff + test suite

python scripts/train_tiny.py data/science-v2/train.txt \
  --validation-text data/science-v2/validation.txt \
  --steps 200 --device cpu --output-dir artifacts/runs/quickstart
python scripts/evaluate.py data/science-v2/train.txt \
  --validation-text data/science-v2/validation.txt --device cpu \
  --checkpoint artifacts/runs/quickstart/best_checkpoint.pt
python scripts/generate.py "Natural selection" --checkpoint artifacts/runs/quickstart/best_checkpoint.pt
```

## Reproducibility

```bash
./scripts/verify_release.sh                  # lint, tests, strict record validation, report drift
python scripts/validate_records.py --strict  # model identities verified from artifacts or the pinned baseline

# published v1 run: legacy backend selected explicitly; bitwise on CPU with torch 2.14 (perplexity 17.58)
python scripts/train_tiny.py data/science/train.txt --validation-text data/science/validation.txt \
  --steps 1000 --eval-interval 250 --gradient-accumulation-steps 4 \
  --device cpu --seed 42 --attention-backend mha --output-dir artifacts/runs/science-v1-s42

PYTHON=python ./experiments/sdpa_v1.sh       # v1: MHA vs SDPA training, 3 seeds
PYTHON=python ./experiments/v2_protocol.sh   # v2: verify corpus, train, select on validation, test once
```

Checkpoint files are not distributed, and they embed elapsed time, so a regenerated file does not reproduce its recorded SHA-256. Models are compared by a deterministic fingerprint instead (`minilm-model-identity/v1`). Historical fingerprints are anchored by an audited baseline whose hash is pinned in code: an integrity anchor in reviewed code, not a cryptographic signature. Rules: [RECORD_SCHEMA.md](docs/RECORD_SCHEMA.md) · checks: [RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Documentation

- [Experiment results](docs/EXPERIMENT_RESULTS.md): all tables, method and limitations
- [Research report](docs/research_report.md) and [paper (PDF, v1 only)](output/pdf/MiniLM-Lab-Research-Paper.pdf)
- [Implementation audit](docs/implementation_audit.md): evidence for each correctness claim
- [Record schema and provenance](docs/RECORD_SCHEMA.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Experiment protocol](docs/experiment_protocol.md) · [Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md)
- Corpus manifests: [v1](data/science/manifest.json) · [v2](data/science-v2/manifest.json)

## Limitations

- **Small, under-trained models.** Every run is 1,000 optimizer steps on about 1 MB of text and spends most of it on an early loss plateau; results are seed-sensitive, and three seeds give a spread, not a confidence interval.
- **One held-out test book.** The v2 test set measures transfer to one unseen author and topic, not broad generalization. Its results are published and can no longer guide tuning under an untouched-test claim.
- **Legacy v1 corpus.** 26% of v1 held-out bytes are Darwin's alphabetical index, and v1 validation both selects checkpoints and is reported; v2 fixes both.
- **One machine.** Timings come from one laptop; MPS findings reflect a tiny model where launch overhead dominates; CUDA is untested; no cross-device or cross-version bitwise reproducibility is claimed. The September KV-cache speedup (3.58×) re-measures at 3.40× on the same hardware.
- **Trust anchor.** Without the original checkpoints, historical model fingerprints rest on the pinned audited baseline.

MIT licensed. © Anvit Devadiga.
