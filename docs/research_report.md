# MiniLM Lab: science prose systems study

**Author:** Anvit Devadiga  
**Run:** one seed, 1,000 optimizer updates, CPU
**Status:** reproducible baseline, not a broad-language benchmark

## Question

Can a compact decoder-only model learn held-out scientific prose while exposing measurable inference and storage trade-offs on laptop-class hardware?

## Data and split

Two Project Gutenberg public-domain editions are used: [Darwin, *On the Origin of Species*](https://www.gutenberg.org/ebooks/1228) and [Einstein, *Relativity: The Special and General Theory*](https://www.gutenberg.org/ebooks/30155). `prepare_science_corpus.py` removes the Gutenberg boilerplate, takes approximately the first 90% of each book's bytes at a paragraph boundary for training, and holds out the remainder. The exact downloaded and prepared SHA-256 hashes are in [the manifest](../data/science/manifest.json). The prepared corpus contains 1,008,977 training bytes and 112,112 validation bytes.

This is a within-document holdout. It tests continuation on unseen passages from the same books, not transfer to unseen authors, topics or factual question answering. The corpus is modest and may contain repeated phrases across splits.

## Model and protocol

Four decoder blocks, width 192, six attention heads, context 128, byte vocabulary 256, learned positions, RMSNorm, SwiGLU MLP, tied embeddings. AdamW with learning rate 3e-4, microbatch size eight, four microbatches per optimizer step, seed 42. Checkpoints are selected by deterministic full-split validation at steps 250, 500, 750 and 1,000. The evaluator scores every next token once in nonoverlapping 128-token chunks, with the first token of each chunk as context. This is a fixed-window metric, not a rolling-context metric.

## Results

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

### Portable INT8 reference

The old dynamic INT8 path failed on this Mac with `NoQEngine`. A portable per-output-channel INT8 reference now quantizes MLP linear weights and dequantizes during each forward pass. Attention projections, embeddings and the tied LM head remain FP32.

| Measure | FP32 | Weight-only INT8 reference |
| --- | ---: | ---: |
| Stored model tensor bytes | 9,844,480 | 4,563,712 |
| Held-out loss | 2.866917 | 2.867067 |
| Median of 32 forward passes | 55.6 ms | 65.9 ms |

The format saves 53.6% of stored tensor bytes in memory. It is slower because dequantization happens at runtime. Actual checkpoint file size, peak memory, and accelerated INT8 kernels were not measured.

## Threats to validity

- Only one seed and one model shape were run. Do not treat the difference between methods as statistically established.
- The science corpus is 1 MB. Perplexity 17.58 does not imply scientific understanding or reliable factual responses.
- CPU timing is local to one execution environment. Apple M4 MPS throughput remains unmeasured in this run.
- The training curves have four validation points. They show checkpoints, not a continuous trajectory.
- The BPE tokenizer and LoRA wrapper remain educational components without a matched end-to-end comparison in this study.

## Reproduce and extend

Run the commands in [README](../README.md). `scripts/publish_science_results.py` checks the local artifacts and writes the tracked JSON record. `scripts/build_report_assets.py` renders the repository figures from that record. The next useful experiments are matched-seed RoPE and learned-position ablations, multi-seed runs, a genuinely unseen-book test, and an M4 MPS benchmark.
