# Release checklist · first research-grade release candidate

Branch `experiments/sdpa-training-v2`, stacked on `experiments/attention-kv-audit`, which
branches from `origin/main` at `1fb8f37`. Not pushed, merged, tagged for release or
published. The local tag `sdpa-v1-frozen` (→ `827b605`) marks the frozen SDPA-on-v1
results. Verified on 2026-10-22 on an Apple M4 (16 GB), macOS 27.0.1, Python 3.12.14,
torch 2.14.1, in the development worktree and in a fresh clone installed from
`requirements/release-lock.txt`.

## Independent audit of `c256f20`: findings and status

| Finding | Status | Fix | Regression tests |
| --- | --- | --- | --- |
| 1. Checkpoint file hashes change on identical retraining | **Resolved** | deterministic model identity `minilm-model-identity/v1` (`src/minilm_lab/identity.py`); file SHA-256 kept for artifact identity; 24 historical checkpoints attested | `tests/test_model_identity.py` (14), attestation tests in `tests/test_record_provenance.py` |
| 2. Legacy `auto` / `mha` duplicates inflate seed counts | **Resolved** | experiment identity after backend canonicalization; replicate / conflict / supersession rules (`src/minilm_lab/records.py`) | `tests/test_record_provenance.py` |
| 3. Evaluation provenance not validated; report joined on seed/backend | **Resolved** | evaluation rows resolve to one experiment identity by checkpoint SHA-256 with seed, backend, protocol, step, corpus, model identity, split, commit and manifest checks; report fails closed | `tests/test_record_provenance.py` |
| 4. KV-cache records lack raw samples | **Resolved for current records** | raw per-repeat samples persisted, summaries derived and re-checked; historical records superseded, not edited | `tests/test_record_provenance.py` |
| A. float64 oracle overstated | Resolved | qualified everywhere (RMSNorm variance and RoPE angles run in float32) | documentation |
| B. CUDA memory could be host RSS | Resolved | `provenance.measure_peak_memory` (CUDA: device allocator peak) | `tests/test_memory_measurement.py` (CUDA stubbed) |
| C. Stale INT8 / corpus statements | Resolved | annotated in `implementation_audit.md`; updated in `research_report.md` | `tests/test_published_claims.py` |
| D. Exit codes through pipelines | Resolved | validator exits 1 on failure; `scripts/verify_release.sh` uses `set -euo pipefail` | `test_validator_failure_propagates_through_exit_codes_and_pipelines` |
| E. No unrelated reformatting | Respected | no formatter run on legacy files | — |

Detailed rules: [RECORD_SCHEMA.md](RECORD_SCHEMA.md).

## Validation commands and results

| Command | Exit | Result |
| --- | ---: | --- |
| `./scripts/verify_release.sh` (development worktree) | 0 | lint clean; 150 passed, 0 failed, 0 skipped; 41 records valid; report regenerated with no documentation drift |
| `./scripts/verify_release.sh` (fresh clone of `d456295`, lock-file install) | 0 | same |
| `python -m ruff check .` | 0 | clean |
| `python -m pytest -q` | 0 | 150 passed |
| `python scripts/validate_records.py` | 0 | `validated 41 records: OK` |
| `python scripts/build_experiment_report.py && git diff --exit-code -- docs/EXPERIMENT_RESULTS.md docs/figures` | 0 | no drift |
| README v1 training and `evaluate.py --attention-backend mha` (fresh clone) | 0 | four checkpoint losses bitwise equal to the published record; perplexity 17.582732137354572 (published value) |
| Regenerated v1 seed-42 checkpoint vs attested original | — | file SHA-256 **differs**; stored model identity matches its tensors; model identity **equals** the attested original |
| `python scripts/prepare_science_corpus.py --output-dir …` (network) | 0 | v1 splits verified against published hashes |
| `python scripts/prepare_science_corpus_v2.py --output-dir …` (network) | 0 | all five v2 files byte-identical to the committed corpus |

Auditor reproductions, rerun on the fixed code (`validate_records.py --records-dir <corrupted copy>`; report via `records.protocol_rows`):

| Scenario | Validator exit | Report |
| --- | ---: | --- |
| unmodified records | 0 | builds; v1 MHA seeds [42, 43, 44] |
| evaluation checkpoint SHA-256 = 64 zeros | 1 | refused |
| evaluation rows swapped to the wrong training runs | 1 | refused |
| evaluation protocol changed | 1 | refused |
| evaluation corpus (test split) hash changed | 1 | refused |
| invalid training commit | 1 | refused |
| invalid manifest hash | 1 | refused |
| legacy `auto` + canonical `mha` seed-42 replicate | 0 | builds; seeds [42, 43, 44] (not inflated) |
| contradictory duplicate seed-42 record | 1 | refused |
| KV summary not derivable from raw samples | 1 | refused |

Before the fix, on `c256f20`: the six corruption scenarios added no validator errors, and the replicate scenario produced MHA seeds [42, 42, 43, 44].

## Published-number changes

Only the KV-cache timings changed, because the replacement records are new measurements. Every other generated section has numbers identical to `c256f20`. KV-cache memory is identical too.

| Claim | Before (historical record) | Now (record with raw samples) |
| --- | --- | --- |
| MPS fp32, batch 1, 1–16-token prompts | 0.87–0.88× | 0.88× |
| CPU cache speedup vs uncached SDPA | batch 1: 2.2–16×; batch 8: 3.8–23× | batch 1: 2.2–15×; batch 8: 3.7–23× |
| CPU vs uncached legacy MHA | up to 62× | up to 63× |
| MPS fp32 at 1,024 tokens, batch 1 | 3.4× | 3.6× |
| MPS past the context window | ~1.0× | 0.94–1.00× |
| README headline "2.2–23× on CPU" | unchanged | unchanged |

Corrected statements: a regenerated checkpoint does not reproduce its file hash (the README previously implied it could be checked against it). The float64 oracle is qualified. The research report's INT8 and corpus-regeneration statements are updated.

## Historical data compatibility

- No record was edited or deleted. 41 records: 14 superseded (listed with reasons and replacements in `experiments/records/SUPERSEDED.json`), 27 current.
- Historical training and evaluation records (schema v1) validate under the documented legacy rules: hyperparameters are parsed from the recorded command, and nothing missing is fabricated. Their model identities come from attestations of byte-verified checkpoint files.
- Original backend labels (`auto`) remain in records; canonicalization happens only when reading.
- Legacy checkpoints (no `model_identity`, no `attention_backend`, older config fields) load and hash.
- The v1 corpus, v1 manifest, published record `docs/data/science_run.json` and tag `sdpa-v1-frozen` are unchanged; the original `MiniLM-Lab` worktree is untouched.

## Reproducibility statement

- Bitwise reproduction of losses and model identity is verified on the same machine type, CPU, Python 3.12.14 and torch 2.14.1. No cross-device or cross-version bitwise reproducibility is claimed.
- Checkpoint **files** are not reproducible byte-for-byte (they embed elapsed seconds). Use the model identity to check a regenerated model.
- Timings are environment-dependent measurements with medians, p95 and CV derived from stored raw samples; they are not expected to reproduce exactly.

## Test-set status

The Faraday test set was evaluated exactly once, after all training (`v2_final_cpu`). Its results are published and must not guide further tuning under the same untouched-test claim. The v2 cleaning rules were designed while viewing its raw text (structure only, never model output). One book does not establish broad cross-author generalization.

## Outstanding limitations (not blocking)

| Item | Status |
| --- | --- |
| Historical KV raw samples | lost for the three superseded records; current results use re-measurements |
| Model identity scope | byte-tokenizer checkpoints only (BPE checkpoints raise) |
| Float64 oracle | RMSNorm variance and RoPE angles still float32 (documented, not changed) |
| CUDA | untested; CUDA memory path tested with stubbed counters only |
| MPS training determinism | not verified; all published training ran on CPU |
| Not re-verified | BPE tokenizer training path, LoRA integration, RoPE ablation |
| Code formatting | `ruff format` is not a gate; legacy files not reformatted |
| Local paths in historical records | five records contain absolute local paths; append-only, left as written |
| Report-drift test | fails by design while `docs/` has uncommitted edits |
| Older artifacts | PDF paper and September figures describe single-seed v1 only |
| Release actions | push, merge, release tag and public release await approval |
