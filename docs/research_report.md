# MiniLM Lab: science prose systems study

**Author:** Anvit Devadiga  
**Status:** reproducible small-model study, not a broad-language benchmark  
**Protocols:** science-v1 (legacy, September 2026) and science-v2 (October 2026). Perplexities are not comparable across protocols.

## October 2026 update

All numbers below come from JSON records in `experiments/records/`; the full
generated tables, method and limitations are in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md),
and the evidence for every correctness claim is in the [implementation audit](implementation_audit.md).
Hardware: Apple M4 (16 GB), CPU and MPS, torch 2.14.1. CUDA was not available.

### Follow-ups on science-v1 (corpus unchanged)

- **Reproduction.** The published seed-42 run reproduces bit-for-bit. Seeds 42/43/44
  give held-out perplexity 17.58 / 18.99 / 21.07 (19.21 ± 1.75); the published value
  is the best of three. Every seed sits near 3.27 nats until roughly step 750.
- **Attention implementations.** The legacy `nn.MultiheadAttention` path (`mha`), PyTorch
  SDPA and an explicit reference compute the same function (identical fp32 perplexity,
  100% top-1 agreement with a float64 reference run whose RMSNorm variance and RoPE
  angles are still computed in float32; gradients verified with `gradcheck`).
  At inference SDPA is up to 3.9× (CPU) and 5.4× (MPS) faster than the reference at
  2,048 tokens and needs 8–22× less memory.
- **Matched training, legacy MHA vs SDPA (3 seeds).** Validation perplexity
  19.21 ± 1.75 vs 19.04 ± 1.37 with no consistent paired direction. A controlled
  training-step benchmark shows SDPA 5–28% faster on CPU (L=128–1,024) and within
  ~5% of the other backends on MPS.
- **KV cache.** 2.2–23× faster than recomputation on CPU, but slower than
  recomputation on MPS for batch-1 prompts under 128 tokens, and no faster once
  generation passes the context window (the cache is rebuilt every step).
- The September CPU-only cache speedup (3.58×) and INT8 numbers below are retained
  unchanged. In October both were re-run with the legacy `mha` backend
  (`experiments/records/legacy/`): the INT8 validation losses and stored byte counts
  reproduce exactly, while timings differ (cache speedup 3.40× vs 3.58×; INT8 32-pass
  latency 55.3/65.7 ms vs 55.6/65.9 ms), so the timings are measurements, not constants.
- **Corpus regeneration.** Project Gutenberg changed the Darwin file's header after
  September; `prepare_science_corpus.py` now verifies the regenerated v1 splits against
  the published hashes (byte-identical as of October 22) instead of the raw-file hash.

### Protocol science-v2

**Data.** The same two books, cleaned before splitting by deterministic rules
(exact start/end heading markers per book; headings, fully bracketed editorial
paragraphs and multi-line numeric lists/tables removed; displayed equations kept),
plus a third book by a different author, Faraday's *The Chemical History of a Candle*
(Gutenberg 14474), held out entirely as a test set. Extracted book bodies are pinned by
SHA-256 in `scripts/prepare_science_corpus_v2.py`; split hashes, dropped-paragraph counts
and contamination checks are in [the v2 manifest](../data/science-v2/manifest.json).
No validation or test paragraph is an exact or ≥ 50% word-8-gram duplicate of training
(or validation) text. Train 969,506 bytes; validation 105,666; test 211,849.

**Protocol.** v1 hyperparameters unchanged (nothing tuned on v2); best checkpoint
chosen on validation; test scored once per selected checkpoint after all training.

| Backend | Seeds | Validation perplexity | Test perplexity (Faraday) |
| --- | ---: | ---: | ---: |
| Legacy MHA | 42, 43, 44 | 15.41 ± 2.70 | 15.66 ± 2.58 |
| SDPA | 42, 43, 44 | 15.37 ± 2.73 | 15.60 ± 2.63 |

Test perplexity is 0.1–0.4 above validation in every run: the model transfers to an
unseen author and topic at close to its in-domain level, at this (low) level of
quality. Seeds, not backends, dominate the spread. **These numbers must not be
compared with the v1 perplexities above or below**: v2 removes non-prose text and
uses a different validation set.

The Faraday test set was evaluated once, after all training, and is now published:
it must not guide further tuning under the same untouched-test claim, and one book
does not establish broad cross-author generalization.

From the first release SDPA is the default backend; every v1 reproduction command
passes `--attention-backend mha` (recorded as `auto` before the release).

### Reproduce (October 2026)

```bash
PYTHON=python ./experiments/sdpa_v1.sh        # science-v1 matched training, 3 seeds x 2 backends
PYTHON=python ./experiments/v2_protocol.sh    # build/verify science-v2, train, select, test once
python scripts/benchmark_attention.py --device cpu   # and --device mps [--dtype float16]
python scripts/benchmark_kv_scaling.py --device cpu  # and --device mps [--dtype float16]
python scripts/benchmark_training_step.py --device cpu   # and --device mps
python scripts/build_experiment_report.py
```

### Remaining limitations

Under-trained models (1,000 steps) whose quality depends strongly on when each seed
leaves its early plateau; three seeds; one laptop; a single test book; no CUDA.
The PDF paper is generated from the September science-v1 record only and does not
include this update.

## Legacy baseline · science-v1 (September 2026, single seed)

The section below is the original report, unchanged. Its "next experiments" list is
partly addressed above (multiple seeds, unseen-book test, M4 MPS measurements); the
RoPE ablation remains open.

### Question

Can a compact decoder-only model learn held-out scientific prose while exposing measurable inference and storage trade-offs on laptop-class hardware?

### Data and split

Two Project Gutenberg public-domain editions are used: [Darwin, *On the Origin of Species*](https://www.gutenberg.org/ebooks/1228) and [Einstein, *Relativity: The Special and General Theory*](https://www.gutenberg.org/ebooks/30155). `prepare_science_corpus.py` removes the Gutenberg boilerplate, takes approximately the first 90% of each book's bytes at a paragraph boundary for training, and holds out the remainder. The exact downloaded and prepared SHA-256 hashes are in [the manifest](../data/science/manifest.json). The prepared corpus contains 1,008,977 training bytes and 112,112 validation bytes.

This is a within-document holdout. It tests continuation on unseen passages from the same books, not transfer to unseen authors, topics or factual question answering. The corpus is modest and may contain repeated phrases across splits.

### Model and protocol

Four decoder blocks, width 192, six attention heads, context 128, byte vocabulary 256, learned positions, RMSNorm, SwiGLU MLP, tied embeddings. AdamW with learning rate 3e-4, microbatch size eight, four microbatches per optimizer step, seed 42. Checkpoints are selected by deterministic full-split validation at steps 250, 500, 750 and 1,000. The evaluator scores every next token once in nonoverlapping 128-token chunks, with the first token of each chunk as context. This is a fixed-window metric, not a rolling-context metric.

### Results

| Measure | Value |
| --- | ---: |
| Best checkpoint | step 1,000 |
| Held-out loss | 2.8669 nats/token |
| Held-out perplexity | 17.5827 |
| Bits per byte | 4.1361 |
| Scored targets | 112,111 bytes |
| Uncached 96-token decode | 119.4 ms median |
| KV-cached 96-token decode | 33.4 ms median |
| Cache speedup | 3.58× |

| Held-out source | Scored bytes | Perplexity | Bits per byte |
| --- | ---: | ---: | ---: |
| Darwin | 93,405 | 17.76 | 4.1505 |
| Einstein | 18,703 | 16.68 | 4.0597 |

Darwin contributes most of the aggregate held-out bytes. Each per-book evaluation starts its own context, so their target counts sum to two fewer than the concatenated evaluation; use the aggregate number for the headline result.

The decode benchmark uses a fixed 21-token prompt, one warm-up and seven timed runs per mode, the same sampling seed, temperature, and no top-k filtering. It ran on CPU. The [raw record](data/science_run.json) includes every timing sample and training checkpoint. The [learning](figures/learning.svg), [speed](figures/speed.svg), and [compression](figures/tradeoff.svg) figures are generated from that record.

#### Portable INT8 reference

The old dynamic INT8 path failed on this Mac with `NoQEngine`. A portable per-output-channel INT8 reference now quantizes MLP linear weights and dequantizes during each forward pass. Attention projections, embeddings and the tied LM head remain FP32.

| Measure | FP32 | Weight-only INT8 reference |
| --- | ---: | ---: |
| Stored model tensor bytes | 9,844,480 | 4,563,712 |
| Held-out loss | 2.866917 | 2.867067 |
| Median of 32 forward passes | 55.6 ms | 65.9 ms |

The format saves 53.6% of stored tensor bytes in memory. It is slower because dequantization happens at runtime. Actual checkpoint file size, peak memory, and accelerated INT8 kernels were not measured.

### Threats to validity

- Only one seed and one model shape were run. Do not treat the difference between methods as statistically established.
- The science corpus is 1 MB. Perplexity 17.58 does not imply scientific understanding or reliable factual responses.
- CPU timing is local to one execution environment. Apple M4 MPS throughput remains unmeasured in this run.
- The training curves have four validation points. They show checkpoints, not a continuous trajectory.
- The BPE tokenizer and LoRA wrapper remain educational components without a matched end-to-end comparison in this study.

### Reproduce and extend

Run the commands in [README](../README.md). `scripts/publish_science_results.py` checks the local artifacts and writes the tracked JSON record. `scripts/build_report_assets.py` renders the repository figures from that record. The next useful experiments are matched-seed RoPE and learned-position ablations, multi-seed runs, a genuinely unseen-book test, and an M4 MPS benchmark.
