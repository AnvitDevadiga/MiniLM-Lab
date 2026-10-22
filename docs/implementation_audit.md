# Implementation audit · October 2026

Audit of `1fb8f37` on branch `experiments/attention-kv-audit`, Apple M4 (16 GB),
macOS 27.0.1, Python 3.12.14, torch 2.14.1. Every "verified" row cites the test
or record that verifies it. Baseline before any change: 16/16 tests passed, ruff clean.

## Verified

| Claim | Evidence |
| --- | --- |
| Published run reproduces bit-for-bit | Retraining seed 42 on CPU gives identical train and validation loss at all four checkpoints (torch 2.14.1 vs published 2.14.0) |
| Attention changes leave the default path unchanged | Seed 42 retrained on the modified code gives a bitwise-identical checkpoint |
| Attention tensor shapes | `test_cached_backends_match_full_forward_in_chunks`, `test_cache_grows_by_one_position_per_decoded_token`: cache is `(batch, heads, positions, head_width)` |
| Causal masking | `test_future_tokens_do_not_change_past_logits`: perturbing tokens ≥ t leaves logits < t unchanged, for all three backends × learned/RoPE |
| Cache mask alignment | `test_causal_mask_is_bottom_right_aligned`; chunked prefill (1, 8, 1, 22 tokens) matches the full forward |
| All backends compute the same function | `test_backends_match_float64_oracle` (< 1e-4 max logit error); trained checkpoints: identical perplexity to 4 decimals in fp32 on CPU and MPS |
| RoPE correctness | matches complex-number rotation exactly; scores depend only on m − n; norm preserved |
| KV cache equivalence | cached and uncached greedy decoding are token-identical for single-token prompts, prompts at the context limit, batch 4, and 40 tokens past the window (sliding refresh) |
| Batch independence | each row of a batched cached decode equals decoding that row alone |
| Gradient accumulation | 2 × 4-sequence microbatches with `loss / 2` reproduce the 8-sequence gradient (`test_gradient_accumulation_matches_full_batch`) |
| Checkpoint resume | 3 + 3 resumed steps give bitwise-identical weights to 6 uninterrupted steps |
| Train/validation separation | separate files whose hashes match the manifest; 1 of 318 validation paragraphs (> 40 chars) also appears in train; 0.04% of 64-character validation windows occur verbatim in train |
| Evaluation counting | every held-out target scored once (existing test) |

## Defects found and fixed

| Defect | Evidence | Fix |
| --- | --- | --- |
| RoPE positions and angles computed in the activation dtype | bf16 max error 0.38 at 128 positions and 7.3 at 2048, versus a 0.008–0.015 rounding floor; fp16 2.8 at 2048 | Angles in float32, cast after `cos`/`sin`; `test_rope_half_precision_error_is_at_rounding_level` |
| `compare_attention_quality.py` (new in this branch) cast MPS tensors to float64 | crash on MPS | Move to CPU before upcasting |
| MPS memory measured via driver pool size (new in this branch) | values quantized to ~1 GiB heaps | Per-op sampling of live allocator bytes; first MPS records kept but superseded |

## Findings not changed (need a decision or are out of scope)

| Finding | Consequence |
| --- | --- |
| Default attention (`nn.MultiheadAttention` with an explicit boolean mask) is used by training, evaluation and uncached generation | On CPU at B=8, L=2048 it is 4.5× slower and uses 8× more memory than SDPA with `is_causal` (see EXPERIMENT_RESULTS.md). Switching the default changes training numerics slightly, so it should be its own experiment |
| 26% of all validation bytes (31.5% of Darwin's) are the book's alphabetical index | Held-out perplexity mixes prose with index entries; per-source numbers are affected most |
| The validation split both selects the best checkpoint and is reported | No untouched test split. The best checkpoint was the final step in all three seeds, so selection bias is currently nil, but the protocol allows it |
| Training plateaus near unigram entropy (≈3.27 nats) until ~750 of 1,000 steps in every seed | The published checkpoint is captured as it leaves the plateau. Seeds 42/43/44 give perplexity 17.58 / 18.99 / 21.07, so the single-seed headline is optimistic |
| Evaluation uses non-overlapping 128-token chunks | Early tokens in each chunk see little context. Perplexity is pessimistic relative to sliding-window evaluation (correctly counted, but not comparable to sliding-window numbers) |
| Resumed runs log an extra evaluation at each segment boundary | Weights are identical; `metrics` gains a row, and `best_checkpoint.pt` could be chosen at a step an uninterrupted run never evaluates |
| RoPE path ignores attention dropout; RoPE models still allocate the learned position table | Inert at the published `dropout=0`; inflates RoPE parameter counts |
| `benchmark_generation.py` has no warmup and one sample; it and `generate.py` load checkpoints with `weights_only=False` | Its single timing is not a benchmark; unsafe deserialization of untrusted checkpoints |

## Unverified assumptions

- Upstream Gutenberg sources were not re-downloaded. The local split files match the manifest hashes, but the raw-source hashes were not re-checked.
- MPS training determinism and resume on MPS (the RNG state saved is CPU-only; batch sampling uses the CPU RNG, dropout would use the device RNG).
- The BPE tokenizer path through training/evaluation, LoRA integration into the model, and the published INT8 numbers were not re-measured in this pass.
- CUDA code paths (`synchronize`, `max_memory_allocated`) are untested: no CUDA hardware.
- Benchmark timings assume a quiet machine. Thermal state and background load were not controlled beyond warmups, rotated backend order and repeat statistics.

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
