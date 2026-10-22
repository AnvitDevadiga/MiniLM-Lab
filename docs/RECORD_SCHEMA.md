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

## Experiment identity and selection (training)

A training record's experiment identity is:

`(protocol, canonical backend, seed, config digest, dataset digest)`

- **canonical backend**: `auto` (pre-release name) and a missing field both mean `mha`. Canonicalization happens before any grouping. Original labels stay in the records.
- **config digest**: SHA-256 of canonical JSON of the full model config, tokenizer and hyperparameters (steps, batch size, gradient accumulation, eval interval, learning rate, optimizer, device).
- **dataset digest**: SHA-256 of canonical JSON of the training and validation file hashes.

Selection, applied in order:

1. Records listed in `SUPERSEDED.json` are excluded. They stay on disk with a reason and a replacement.
2. Records sharing an identity are **replicates** only if their per-step train/validation losses, selected step and model identity (where both have one) are identical. The earliest record (by `created_utc`, then file name) is canonical; the seed is counted once.
3. Records sharing an identity with different outcomes are a **conflict**. Validation and reporting fail until one record is explicitly superseded.

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

The validator recomputes median, p95, IQR, mean, CV, min and max from the raw values (relative tolerance 1e-12). Units and the synchronization strategy are recorded in `settings`. The three KV-cache records from 2026-10-22 03:55–04:00 UTC predate raw samples. They are superseded, kept unchanged, and their samples cannot be recovered.

Peak memory: CPU uses the host peak-RSS delta, MPS uses live allocator bytes sampled after every op, and CUDA uses the device allocator peak (`torch.cuda.max_memory_allocated`). Host RSS is never reported for an accelerator. The CUDA path is untested.

## Supersession

`SUPERSEDED.json` maps a record key to `{reason, superseded_by}`. Both records must exist. Superseded records are never used by reports, but stay on disk as the audit trail.
