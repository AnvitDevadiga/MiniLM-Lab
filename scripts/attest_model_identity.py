"""Attest deterministic model identities for training records that predate them.

Training records written before schema v2 store only checkpoint *file* hashes.
For every current training record whose checkpoints are still on disk, this
script verifies the files are byte-identical to the recorded SHA-256 values and
then records each checkpoint's model identity (minilm-model-identity/v1) in a new,
append-only record. Nothing is inferred for checkpoints that are not available.

--regenerated RECORD_KEY=PATH adds a regeneration check: the identity of a
checkpoint retrained from the record's command, compared with the original.
"""

import argparse
import shlex
from pathlib import Path

import torch

from minilm_lab.identity import SCHEME, checkpoint_identity
from minilm_lab.provenance import environment, file_sha256, write_record
from minilm_lab.records import load_store

ROOT = Path(__file__).resolve().parents[1]


def output_dir(record: dict) -> Path:
    tokens = shlex.split(record["settings"]["training_command"])
    return ROOT / tokens[tokens.index("--output-dir") + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regenerated", action="append", default=[], help="RECORD_KEY=checkpoint path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/records/identity")
    args = parser.parse_args()
    store = load_store(ROOT)
    attested, unavailable = [], []
    for key, record in store.current("training"):
        if record["results"].get("best_model_identity_sha256"):
            continue  # schema v2 records carry their own identity
        directory = output_dir(record)
        for kind, filename in (("best", "best_checkpoint.pt"), ("final", "tiny_checkpoint.pt")):
            path = directory / filename
            expected = record["results"][f"{kind}_checkpoint_sha256"]
            if not path.is_file() or file_sha256(path) != expected:
                unavailable.append({"training_record": key, "kind": kind, "reason": "file missing or not byte-identical to the recorded hash"})
                continue
            identity = checkpoint_identity(torch.load(path, map_location="cpu", weights_only=True))
            attested.append({"training_record": key, "kind": kind, "checkpoint_path": str(path.relative_to(ROOT)),
                             "checkpoint_sha256": expected, "model_identity_sha256": identity["sha256"]})
    regenerated = []
    by_record = {(row["training_record"], row["kind"]): row for row in attested}
    for item in args.regenerated:
        key, path = item.split("=", 1)
        original = by_record[(key, "best")]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = checkpoint_identity(checkpoint)["sha256"]
        regenerated.append({"training_record": key, "regenerated_checkpoint_sha256": file_sha256(Path(path)),
                            "original_checkpoint_sha256": original["checkpoint_sha256"],
                            "file_hash_equal": file_sha256(Path(path)) == original["checkpoint_sha256"],
                            "model_identity_sha256": identity, "model_identity_equal": identity == original["model_identity_sha256"]})
    payload = {
        "experiment": "model_identity",
        "environment": environment("cpu"),
        "settings": {"scheme": SCHEME, "verification": "checkpoint file SHA-256 equals the training record before hashing tensors"},
        "results": {"attested": attested, "unavailable": unavailable, "regeneration_checks": regenerated},
    }
    for row in regenerated:
        print(f"{row['training_record']}: file hash equal={row['file_hash_equal']}, model identity equal={row['model_identity_equal']}")
    print(f"attested {len(attested)} checkpoints, {len(unavailable)} unavailable")
    print(f"wrote {write_record(args.output_dir, 'model_identity_attestation', payload)}")


if __name__ == "__main__":
    main()
