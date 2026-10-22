# Experiment records: schema, identity and provenance rules

Records live in `experiments/records/<kind>/<UTC timestamp>_<name>_<commit7>.json`.
They are append-only: a record is never edited or deleted. The rules below are
implemented in `src/minilm_lab/records.py` and enforced by
`python scripts/validate_records.py` (exit status 1 on any problem) and by
`scripts/build_experiment_report.py`, which refuses to build from invalid records.

## Record versions

| `record_version` | Written | Adds |
| ---: | --- | --- |
| 1 | before commit `cd6a11d` | environment, settings, results; checkpoint file SHA-256 |
| 2 | from commit `cd6a11d` | training: `best_/final_model_identity_sha256`, explicit `run_summary.hyperparameters`; evaluation and quality rows: `model_identity_sha256`; KV-cache benchmarks: raw samples |

Every record has `environment` (Python, torch, platform, processor, device, threads,
git commit and dirty flag, command), `settings` and `results`. A record whose git tree
was dirty must be listed in `SUPERSEDED.json`.

## Two kinds of identity

- **Artifact identity**: the SHA-256 of a checkpoint *file*. It identifies one file. Checkpoints embed elapsed train/eval seconds, so retraining the same model produces a different file hash.
- **Model identity** (`minilm-model-identity/v1`, `src/minilm_lab/identity.py`): SHA-256 of a versioned canonical stream:
  1. `minilm-model-identity/v1\n`
  2. canonical JSON (sorted keys, `,`/`:` separators, ASCII, no NaN) of every `MiniLMConfig` field plus the tokenizer identifier, then `\n`
  3. for each state-dict tensor in ascending name order: canonical JSON header `{"dtype", "name", "nbytes", "shape"}`, then `\n`, then the elements in row-major order, each element little-endian, in the tensor's own dtype (no casting).

  Excluded: elapsed time, metrics, optimizer and RNG state, file paths, timestamps, seeds and the attention backend. Defined for byte-tokenizer checkpoints only. Legacy checkpoints are hashed from their tensors; configs missing newer fields take their defaults.

Historical training records (version 1) have no model identity. `scripts/attest_model_identity.py` verifies that a run's checkpoint files are byte-identical to the recorded file hashes and then records their model identity in `identity/` records. It never infers an identity for a missing file.

## Model-identity resolution and evidence levels

For each current training record, the best and final checkpoints are resolved separately (`records.resolve_identity`). All available sources are collected:

| Source | Where it comes from | Independent of the records? |
| --- | --- | --- |
| `declared` | the training record's `best_/final_model_identity_sha256` (schema v2) | no |
| `attested` | any current `identity/` attestation row with the same checkpoint SHA-256 | no (same mutable record set) |
| `baseline` | the pinned audited baseline entry for that checkpoint SHA-256 | anchored by a pin in code (see below) |
| `artifact` | the checkpoint file: its bytes must hash to the recorded SHA-256, then the identity is recomputed from its tensors and config | yes |

Every available source must give the same fingerprint. A disagreement is a validation failure and leaves the checkpoint with no fingerprint. If a checkpoint file exists at the recorded location but its bytes do not match the recorded SHA-256, validation fails; it does not fall back to other sources.

| Evidence level | Meaning |
| --- | --- |
| `artifact_verified` | the artifact was present, its bytes matched, and the recomputed identity agrees with every other source |
| `trusted_baseline_verified` | no artifact; the pinned audited baseline has an entry and every other source agrees with it |
| `metadata_consistent_only` | only declared and/or attested values, mutually consistent. **Not verification**: a consistent forgery of both is undetectable at this level |
| `unverified` | no fingerprint, or sources disagree |

`validate_records.py` prints the count per level. `--strict`, used by `scripts/verify_release.sh` and by the report builder, requires every current training checkpoint to be `artifact_verified` or `trusted_baseline_verified`. `--require-artifacts` requires `artifact_verified` for all of them.

### Audited baseline and trust assumptions

`experiments/baselines/model_identity_baseline.json` maps each historical checkpoint SHA-256 to its model identity. `scripts/build_identity_baseline.py` produced it on 2026-10-22 from the 24 checkpoint files of the 12 current training runs. It hashed each file, required the result to equal the training record's SHA-256, and recomputed the identity from the tensors; it does not read attestations or declared values. Its output is deterministic: `build_identity_baseline.py --check` regenerates it byte-for-byte wherever the artifacts exist. Its SHA-256 is pinned as `BASELINE_SHA256` in `src/minilm_lab/records.py`. A baseline whose bytes do not match the pin is ignored and reported.

What this does and does not establish:

- It binds historical model identities to a specific, reviewable file whose change requires a code change, so attestations and declared values can no longer silently redefine them.
- It is **not** a cryptographic signature and gives no authenticity against someone who can change both the baseline and the pin. Trust rests on the repository's review history and on the maintainer's statement that the files were checked against the original artifacts, which are not distributed.
- Anyone holding the original artifacts can re-establish `artifact_verified` with `validate_records.py --require-artifacts` and check the baseline with `build_identity_baseline.py --check`.
- New training runs (schema v2) declare their identity, but without their artifact or a baseline update they are only `metadata_consistent_only` and fail `--strict`.

## Experiment identity, duplicates and selection (training)

A training record's experiment identity is:

`(protocol, canonical backend, seed, config digest, dataset digest)`

- **canonical backend**: `auto` (pre-release name) and a missing field both mean `mha`. Canonicalization happens before any grouping. Original labels stay in the records.
- **config digest**: SHA-256 of canonical JSON of the full model config, tokenizer and hyperparameters (steps, batch size, gradient accumulation, eval interval, learning rate, optimizer, device).
- **dataset digest**: SHA-256 of canonical JSON of the training and validation file hashes.

Records are first grouped by **experiment slot** `(protocol, canonical backend, seed, dataset digest)`. Backend canonicalization and model-identity resolution happen before comparison. Selection, applied in order:

1. Records listed in `SUPERSEDED.json` are excluded. They stay on disk with a reason and a replacement.
2. Records in the same slot are **duplicates of one experiment** only if all of the following hold:
   - the configuration digest is the same;
   - per-step train/validation losses and the selected step are identical;
   - checkpoint identity agrees: the same checkpoint file with consistent resolved fingerprints, or different files whose resolved fingerprints are both established and equal.

   The seed is then counted once, and the earliest record represents the group. They are identical, so this choice cannot change any result.
3. Any other combination is a **conflict**: no record of that slot is selected, and validation and reporting fail until an explicit supersession resolves it. Record order or file names never pick between conflicting records.

Every current training record must also name a known protocol. Its training and validation hashes must match the files on disk and the protocol manifest. Its `manifest_path` and `manifest_sha256` (when present) must match the protocol manifest. Its `training_code_commit` must exist in the repository.

**Legacy rule (version 1).** Hyperparameters are not stored explicitly; they are parsed from the recorded `training_command`. Absent flags take `train_tiny.py`'s defaults (`steps 100, batch size 8, accumulation 1, eval interval 25, device auto`), unchanged since `1fb8f37` and guarded by a test.

## Evaluation provenance

Every row of a current evaluation, attention-quality or legacy record must resolve to **exactly one experiment identity** through its checkpoint file SHA-256. Several current training records may share that checkpoint only if they are replicates of one identity. The row must then agree with the training record on:

- protocol (evaluation `settings.protocol` = training `settings.protocol`);
- seed and canonical backend;
- selected step (`selected_step` = training `best_step`);
- corpus hash (`corpus_sha256` = the hash stored in the checkpoint);
- model identity, when the row and the training record (or an attestation) both have one;
- validation split (the evaluation's `validation` file = the file training selected on).

Every evaluation split must lie inside the protocol's data directory, match the file on disk, and match the protocol manifest wherever the manifest lists the file. Rows that map to the same experiment identity twice are rejected. Reports join training and evaluation through these rules only, never by seed and backend alone, and fail if a selected training identity lacks an evaluation row.

| Protocol | Manifest | Data directory |
| --- | --- | --- |
| `science-v1` | `data/science/manifest.json` (train/validation hashes) | `data/science/` |
| `science-v2` | `data/science-v2/manifest.json` (all split files) | `data/science-v2/` |

## Benchmarks and raw samples

Current benchmark records must contain the raw measurements behind every summary:

- attention: `samples_seconds` (forward, prefill, per-step decode);
- training step: `step_seconds_samples`;
- KV cache: `raw.step_seconds`, one list of per-step seconds per measured repeat.

The validator recomputes median, p95, IQR, mean, CV, min and max from the raw values (relative tolerance 1e-12). Every raw timing must be finite and positive.

**Derived KV-cache ratios.** `speedup_cached_vs_uncached_sdpa`, `speedup_cached_vs_uncached_mha` and the pre-release `speedup_cached_vs_uncached_default` are defined as:

`median over repeats of the uncached mode's total time (ms)` ÷ `median over repeats of the cached mode's total time (ms)`

A repeat's total is the sum of its per-step seconds × 1000, prefill included. The ratio is unitless, and > 1 means the cache is faster. The validator derives each ratio from the raw samples (`records.kv_speedups`) and requires any stored ratio to be finite, positive and equal within relative tolerance 1e-12. The report renders only the derived values, never stored ratio fields. The legacy single-prompt KV benchmark (`legacy/`) is checked the same way: medians from its samples, `speedup` = uncached ÷ cached, tokens/s = tokens ÷ median.

**Legacy rule for summary-only KV records.** The three superseded historical KV records have no raw samples. Their summaries are checked for finite, positive times, and their stored ratios must equal the ratio of their stored medians. Their summaries themselves cannot be reconstructed or independently verified. Units and the synchronization strategy are recorded in `settings`. The three KV-cache records from 2026-10-22 03:55–04:00 UTC predate raw samples. They are superseded, kept unchanged, and their samples cannot be recovered.

Peak memory: CPU uses the host peak-RSS delta, MPS uses live allocator bytes sampled after every op, and CUDA uses the device allocator peak (`torch.cuda.max_memory_allocated`). Host RSS is never reported for an accelerator. The CUDA path is untested.

## Supersession

`SUPERSEDED.json` maps a record key to `{reason, superseded_by}`. Both records must exist and be the same kind of record: same seed and canonical backend for training, same device and dtype for benchmarks, same name for evaluations. Entries may not supersede themselves or form cycles, and every chain must end at a current record. Superseded records are never used by reports, but stay on disk as the audit trail.
