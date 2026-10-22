"""Build or check the audited model-identity baseline for historical checkpoints.

For every current training record whose checkpoint files are present, this script
hashes the checkpoint bytes, requires them to equal the SHA-256 recorded in the
training record, and recomputes the model identity from the checkpoint tensors. It
never reads attestation records or declared fingerprints. The output is deterministic
(no timestamps), so anyone holding the original artifacts can regenerate it
byte-for-byte with --check.

After review, the file's SHA-256 is pinned as BASELINE_SHA256 in
src/minilm_lab/records.py; validation trusts the baseline only while that pin matches.
"""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch

from minilm_lab.identity import SCHEME, checkpoint_identity
from minilm_lab.records import BASELINE_PATH, load_store

ROOT = Path(__file__).resolve().parents[1]


def build(artifact_root: Path) -> tuple[bytes, list[str]]:
    store = load_store(ROOT, baseline_path=None)
    entries, missing = {}, []
    for key, record in store.current("training"):
        tokens = shlex.split(record["settings"]["training_command"])
        directory = artifact_root / tokens[tokens.index("--output-dir") + 1]
        for kind, filename in (("best", "best_checkpoint.pt"), ("final", "tiny_checkpoint.pt")):
            expected = record["results"][f"{kind}_checkpoint_sha256"]
            path = directory / filename
            if not path.is_file():
                missing.append(f"{key} {kind}: {path} not found")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                missing.append(f"{key} {kind}: bytes {actual[:12]} do not match recorded {expected[:12]}")
                continue
            identity = checkpoint_identity(torch.load(path, map_location="cpu", weights_only=True))
            entries[expected] = {"kind": kind, "model_identity_sha256": identity["sha256"], "training_record": key}
    baseline = {
        "scheme": SCHEME,
        "method": "checkpoint bytes hashed and required to equal the training record's SHA-256; "
                  "model identity recomputed from the checkpoint tensors and configuration",
        "entries": dict(sorted(entries.items())),
    }
    return (json.dumps(baseline, indent=2, sort_keys=True) + "\n").encode(), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare with the committed baseline")
    args = parser.parse_args()
    data, missing = build(args.artifact_root)
    for line in missing:
        print(f"MISSING {line}")
    if missing:
        sys.exit("baseline incomplete: every current training checkpoint must be present and byte-identical")
    digest = hashlib.sha256(data).hexdigest()
    target = ROOT / BASELINE_PATH
    if args.check:
        same = target.is_file() and target.read_bytes() == data
        print(f"regenerated baseline {digest} {'matches' if same else 'DIFFERS FROM'} {BASELINE_PATH}")
        sys.exit(0 if same else 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"wrote {BASELINE_PATH} ({len(json.loads(data)['entries'])} checkpoints)\nSHA-256 {digest} (pin as BASELINE_SHA256)")


if __name__ == "__main__":
    main()
