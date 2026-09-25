# Implementation audit · September 2026

## Corrected defects

| Defect | Consequence | Fix |
| --- | --- | --- |
| Training documentation counted microbatches as steps | Compute budget was misleading | Each step now performs a full optimizer update after the configured number of microbatches |
| Best checkpoint used one random validation batch | Checkpoint choice varied with RNG | Full held-out split is scored at every checkpoint |
| Bits per byte used the entire input corpus as denominator | Reported quality was wrong | Only scored held-out bytes are counted |
| Evaluation skipped tail targets | Incomplete validation | Every next token is scored once |
| Cache with multi-token continuation exposed future keys | Causal leak | Explicit offset-aligned attention mask and regression test |
| Cache decode performed a forward pass after the final token | Inflated cache latency | Final pass removed |
| Cache benchmark used different sampling filters | Speedup was not an equal comparison | Same sampler settings, warm-up, seven repeats, raw timings |
| Dynamic INT8 command crashed with `NoQEngine` | Advertised feature did not run | Portable weight-only INT8 reference with quality, storage, and latency measurements |
| Repo-derived corpus mixed the model's own code and report | Weak external validity and unstable source | Two cited public-domain science books, per-book held-out paragraphs, SHA-256 manifest |
| README SVG labels collided and mixed unrelated corpus/tokenizer claims | Visually and scientifically misleading | Paper-style plots from a single measured run, rendered and inspected |

## Remaining limits

The model still delegates multi-head attention, tensor operations and differentiation to PyTorch. BPE and LoRA are building blocks without a matched comparison here. One seed, one small corpus, one CPU environment, and a 128-token context do not establish state-of-the-art quality. The next rigorous steps are a truly unseen-book test, multiple seeds, M4 MPS measurements, and a matched RoPE ablation.
