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
| All backends compute the same function | `test_backends_match_float64_oracle` (< 1e-4 max logit error); trained checkpoints: identical perplexity to 4 decimals in fp32 on CPU and MPS **Update (release audit):** the float64 oracle still computes RMSNorm variance and RoPE angles in float32. |
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
| Default attention (`nn.MultiheadAttention` with an explicit boolean mask) is used by training, evaluation and uncached generation | On CPU at B=8, L=2048 it is 4.5× slower and uses 8× more memory than SDPA with `is_causal` (see EXPERIMENT_RESULTS.md). Switching the default changes training numerics slightly, so it should be its own experiment **Update:** SDPA is now a selectable training backend; matched experiments in EXPERIMENT_RESULTS.md (Experiment 04, Part B) support making it the default, which is left as an explicit decision |
| 26% of all validation bytes (31.5% of Darwin's) are the book's alphabetical index | Held-out perplexity mixes prose with index entries; per-source numbers are affected most **Update:** unchanged in legacy science-v1; removed in science-v2 |
| The validation split both selects the best checkpoint and is reported | No untouched test split. The best checkpoint was the final step in all three seeds, so selection bias is currently nil, but the protocol allows it **Update:** science-v2 adds an untouched test book (Faraday), evaluated once per selected checkpoint |
| Training plateaus near unigram entropy (≈3.27 nats) until ~750 of 1,000 steps in every seed | The published checkpoint is captured as it leaves the plateau. Seeds 42/43/44 give perplexity 17.58 / 18.99 / 21.07, so the single-seed headline is optimistic |
| Evaluation uses non-overlapping 128-token chunks | Early tokens in each chunk see little context. Perplexity is pessimistic relative to sliding-window evaluation (correctly counted, but not comparable to sliding-window numbers) |
| Resumed runs log an extra evaluation at each segment boundary | Weights are identical; `metrics` gains a row, and `best_checkpoint.pt` could be chosen at a step an uninterrupted run never evaluates |
| RoPE path ignores attention dropout; RoPE models still allocate the learned position table | Inert at the published `dropout=0`; inflates RoPE parameter counts |
| `benchmark_generation.py` has no warmup and one sample; it and `generate.py` load checkpoints with `weights_only=False` | Its single timing is not a benchmark; unsafe deserialization of untrusted checkpoints |

## Unverified assumptions

- Upstream Gutenberg sources were not re-downloaded. The local split files match the manifest hashes, but the raw-source hashes were not re-checked. **Update:** re-downloaded on 2026-10-22; the Darwin raw hash changed, but both v1 splits regenerate byte-identically.
- MPS training determinism and resume on MPS (the RNG state saved is CPU-only; batch sampling uses the CPU RNG, dropout would use the device RNG).
- The BPE tokenizer path through training/evaluation, LoRA integration into the model, and the published INT8 numbers were not re-measured in this pass. **Update:** the INT8 losses and stored bytes were re-verified exactly in October (`experiments/records/legacy/`); BPE and LoRA remain unverified.
- CUDA code paths (`synchronize`, `max_memory_allocated`) are untested: no CUDA hardware.
- Benchmark timings assume a quiet machine. Thermal state and background load were not controlled beyond warmups, rotated backend order and repeat statistics.

## Independent release audit · resolution (October 2026)

An independent audit of `c256f20` raised four medium-severity findings. Each was
reproduced before fixing; regression tests cover every reproduction.

| Finding | Reproduction on `c256f20` | Resolution |
| --- | --- | --- |
| 1. Checkpoint file hashes are not reproducible | v1 seed 42 retrained in a fresh clone: identical tensors, config and losses; different file SHA-256 (elapsed/train/eval seconds in metadata) | Deterministic model identity `minilm-model-identity/v1` (`src/minilm_lab/identity.py`), stored in new checkpoints and schema-v2 records; file SHA-256 kept as artifact identity; 24 historical checkpoints attested after verifying their file hashes. The regenerated seed-42 file has a different hash and the same identity |
| 2. Legacy `auto` and `mha` could double-count a seed | an `mha` re-record of seed 42 made the v1 MHA aggregate use seeds [42, 42, 43, 44] | Experiment identity (protocol, canonical backend, seed, config digest, dataset digest) computed after canonicalization; replicates with identical outcomes count once; conflicting identities fail validation and reporting unless explicitly superseded |
| 3. Evaluation provenance was not validated | zeroed checkpoint hashes, rows swapped across seeds, a changed protocol and duplicate records all passed validation; the report joined on seed/backend | Evaluation, quality and legacy rows must resolve to exactly one experiment identity by checkpoint SHA-256 with matching seed, backend, protocol, selected step, corpus hash and model identity; splits must match files and protocol manifests; training commit and manifest references must exist. The report builder uses the same join and fails closed |
| 4. KV-cache records lacked raw samples | records kept only summaries (`count: 5`) | Raw per-repeat, per-step seconds persisted; summaries derived from them and re-checked by the validator; three historical KV records superseded (not edited) by same-settings re-measurements |

Additional corrections: the float64 oracle claim is qualified (RMSNorm variance and
RoPE angles run in float32); CUDA peak memory now uses the device allocator, never host
RSS (`provenance.measure_peak_memory`); stale INT8 and corpus-regeneration statements are
annotated above; validator failures propagate as exit status 1, including through
`pipefail` pipelines. Rules: [RECORD_SCHEMA.md](RECORD_SCHEMA.md).

## Addendum · SDPA training and corpus v2 (October 2026)

| Finding | Evidence | Resolution |
| --- | --- | --- |
| `reference_causal_attention` downcast float64 scores to float32 (`.float()`) | `torch.autograd.gradcheck` failed for the reference backend only | Promote to at least float32, never below the input dtype; float64 reference now equals SDPA to 1e-12. The quality record that used the float64 "oracle" is superseded; re-running with the corrected oracle changed fp32 logit errors by < 1e-6 (e.g. 3.65e-5 → 3.60e-5) and no conclusion |
| SDPA and reference paths skipped attention dropout in training | code inspection | Apply `nn.MultiheadAttention`'s dropout probability in training mode; inert at the published `dropout=0` |
| Upstream Darwin file changed (`ededa9c0…` → `362142cb…`) | fresh download | The extracted v1 split regenerates byte-identically from the new file; only the Gutenberg header/licence changed. `prepare_science_corpus.py` (v1) still pins the old raw hash and therefore refuses to run; it is left unchanged as a legacy script. v2 pins the hash of the extracted body instead **Update:** the v1 script now verifies regenerated split hashes and runs again. |
| One training record was written while unrelated documentation edits were uncommitted | record `git.dirty = true` | Re-recorded from a clean tree; the dirty record is kept and marked superseded |

**Release note.** From the first release the default backend is `sdpa`. The path called
"default" or `auto` in the statements above is now the explicit backend `mha`; `auto` is
accepted as a legacy alias and checkpoints without a backend field are treated as `mha`.

Verified after the changes:

- Default-backend (`auto`) training on v1 is bitwise identical to the original runs for seeds 42, 43 and 44.
- Per-backend gradients agree with a float64 oracle to a relative 1e-4 for learned and RoPE models; `gradcheck` passes for the attention layer under every backend. (The oracle's RMSNorm variance and RoPE angles run in float32.)
- Training resumes refuse a backend change; checkpoints record the backend.
- v2 corpus: body hashes verified on a fresh download; zero exact and near duplicates across train/validation/test; legacy v1 files still match their original manifest (all enforced by `tests/test_corpus_data.py`).

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
