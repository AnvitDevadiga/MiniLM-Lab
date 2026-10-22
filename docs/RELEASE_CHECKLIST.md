# Release checklist · first research-grade release candidate

Branch `experiments/sdpa-training-v2` (stacked on `experiments/attention-kv-audit`,
which branches from `origin/main` at `1fb8f37`). Not pushed, merged, tagged for release
or published. Local annotated tag `sdpa-v1-frozen` marks the frozen SDPA-on-v1 results.
Verified on 2026-10-22 on an Apple M4 (16 GB), macOS 27.0.1, Python 3.12.14, torch 2.14.1.

## Validation commands

Run from the repository root in a virtual environment built from
`requirements/release-lock.txt` (`pip install -r requirements/release-lock.txt && pip install -e . --no-deps`).

```bash
ruff check .                                   # lint (the project gate)
pytest -q                                      # full suite; MPS tests run where MPS is available
python scripts/validate_records.py             # provenance of every experiment record
python scripts/build_experiment_report.py && git diff --exit-code docs/   # tables/figures match records
# README smoke test (fresh clone, ~3 minutes on an M4 CPU)
python scripts/train_tiny.py data/science/train.txt --validation-text data/science/validation.txt \
  --steps 1000 --eval-interval 250 --gradient-accumulation-steps 4 \
  --device cpu --seed 42 --attention-backend mha --output-dir artifacts/runs/science-v1-s42
python scripts/evaluate.py data/science/train.txt --validation-text data/science/validation.txt \
  --device cpu --attention-backend mha --checkpoint artifacts/runs/science-v1-s42/best_checkpoint.pt
python scripts/prepare_science_corpus.py --output-dir /tmp/v1      # needs network
python scripts/prepare_science_corpus_v2.py --output-dir /tmp/v2   # needs network; compare hashes with data/science-v2
```

## Passing checks

| Area | Check | Result |
| --- | --- | --- |
| Tests | Full suite (`pytest -q`) in the development worktree and in a fresh clone installed from the lock file | 109 passed, 0 failed, 0 skipped |
| Lint | `ruff check .` | clean |
| Attention default | SDPA is the default for model, `train_tiny.py`, `evaluate.py`, `generate.py`, benchmarks | `tests/test_release_backends.py` |
| Explicit backends | `mha`, `reference`, `sdpa` selectable; legacy name `auto` maps to `mha`; unknown names rejected | `tests/test_release_backends.py` |
| Checkpoint compatibility | identical `state_dict` across backends; checkpoints without a backend field resume only as `mha`; backend changes on resume refused; canonical names stored | `tests/test_release_backends.py`, `tests/test_decoding_and_training.py` |
| Causal masking | future-token perturbation per backend (learned, RoPE, batch 2–3); cache mask alignment | `tests/test_attention_backends.py`, `tests/test_release_backends.py` |
| Evaluation correctness | `evaluate_tokens` equals a manual chunked loss for every backend; every target scored once; `evaluate.py` backend handling | `tests/test_release_backends.py`, `tests/test_metrics.py` |
| Numerical tolerances | max absolute logit error vs a float64 oracle: < 1e-10 (CPU fp64), < 1e-4 (CPU fp32), < 1e-3 (MPS fp32), < 5e-2 (MPS fp16); gradients vs fp64 oracle (relative 1e-4); `gradcheck` for every backend | `tests/test_release_backends.py`, `tests/test_attention_backends.py` |
| SDPA dropout | `dropout_p == 0` for every SDPA call in eval mode (forward and cached); applied in training | `test_sdpa_dropout_is_disabled_in_evaluation` |
| v1 bitwise reproduction | README v1 command in a fresh clone reproduces all four published checkpoints' train/validation losses exactly; `evaluate.py --attention-backend mha` gives 17.582732137354572 (published value) | smoke test |
| v1 corpus recoverability | committed v1 files match the v1 manifest; per-book files concatenate to `validation.txt`; regeneration from today's Gutenberg files is byte-identical although Darwin's raw hash changed | `tests/test_corpus_data.py`, `prepare_science_corpus.py` |
| v2 corpus | body hashes verified on download; regenerated files byte-identical to committed; zero exact/near duplicates across train/validation/test; no index/backmatter markers | `tests/test_corpus_data.py`, smoke test |
| Records | 36 records: naming, provenance fields, commit exists, data hashes match committed files, only superseded records dirty, superseded records unused by the report | `scripts/validate_records.py`, `tests/test_records.py` |
| Generated report | rebuilding tables and figures from records leaves `docs/` unchanged | `tests/test_records.py` |
| Published values | README headline numbers recomputed from records (attention speedups and memory, training quality and step speed, KV cache, seeds, v2 test, legacy INT8) | `tests/test_published_claims.py` |
| Legacy INT8 | validation losses (2.866917291863034 / 2.867066869126566) and stored bytes (9,844,480 / 4,563,712) reproduce exactly | `experiments/records/legacy/`, `tests/test_published_claims.py` |
| Test protocol | Faraday test file evaluated once, after all training (`v2_final_cpu` record, 6 checkpoints); referenced by no training command | record audit (see below) |
| Hygiene | no credentials, checkpoints, binaries or caches tracked; all branch changes belong to this work; commits authored by the configured identity | `git log -p origin/main..HEAD` scan |

## Corrections made during release verification

- README and docs said the CPU training-step gain was 8–28%; the record gives 5.2% at L=256, so the stated range is now 5–28%.
- The September KV-cache speedup (3.58×) is a timing. Re-measured on the same hardware with the legacy backend it is 3.40×; the README now cites the re-measurement record and does not present 3.58× as reproducible.
- `scripts/prepare_science_corpus.py` no longer refuses to run on Gutenberg header changes; it verifies the regenerated splits against the published hashes instead and never rewrites the historical manifest.

## Superseded records

Ten records are kept but superseded (listed with reasons and replacements in
`experiments/records/SUPERSEDED.json`): two MPS attention records with pool-size memory,
two attention-quality records (one dirty, one with a float32-softmax oracle), four
training records written with a dirty tree, and two legacy re-measurements written with
uncommitted documentation. None is used by any published table.

## Test-set status

The Faraday test set was evaluated exactly once, by `experiments/v2_protocol.sh` after all
training. Its results are published and must not guide further tuning under the same
untouched-test claim. Disclosure: the v2 cleaning rules were designed while viewing the
raw test book (structure only, never model output).

## Reproducibility statement

- Bitwise reproduction is verified only on the same machine type, CPU, Python 3.12.14 and
  torch 2.14.1. No cross-device or cross-version bitwise reproducibility is claimed;
  MPS and other versions should agree within the tolerances above.
- Checkpoints are not committed. Each training record contains the exact command, seed,
  data hashes and the SHA-256 of the selected and final checkpoints, so checkpoints can be
  regenerated and checked against the record.
- Timing measurements (all benchmarks) are environment-dependent and are reported with
  median, p95 and CV; they are not expected to reproduce exactly.

## Outstanding items (not blocking this candidate)

| Item | Status |
| --- | --- |
| Code formatting | `ruff format` is not a project gate; 43 files would be reformatted (24 already on `origin/main`). Not changed, to avoid an unrelated diff. |
| Local paths in records | three attention-quality records and two legacy INT8 records contain absolute local paths (`/Users/…`, a temporary output path). Records are append-only, so they are left as written; new records store repo-relative paths. |
| Report-drift test | `test_generated_report_matches_records` rebuilds `docs/` and compares with git, so it fails while `docs/` has uncommitted edits (by design). |
| CUDA | untested (no hardware); `synchronize` and memory paths for CUDA are unexercised. |
| MPS training determinism | not verified; all published training ran on CPU. |
| Not re-verified | BPE tokenizer training path, LoRA integration, RoPE ablation (the roadmap item remains open). |
| Older artifacts | the PDF paper and September figures describe single-seed v1 only. |
| Release actions | push, merge, release tag and public release await approval. |
