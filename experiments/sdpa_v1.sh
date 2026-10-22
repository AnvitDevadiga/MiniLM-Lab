#!/usr/bin/env bash
# Matched training/evaluation comparison of the legacy attention path (mha,
# nn.MultiheadAttention; recorded as "auto" before the first release) and SDPA
# on the unchanged science-v1 corpus. Backends are always passed explicitly.
# Identical hyperparameters, seeds, data and hardware (CPU, float32); only
# --attention-backend differs. Backend order alternates per seed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUNS=artifacts/runs/sdpa-v1
COMMIT=$(git rev-parse --short HEAD)
train() {  # seed backend
  local out=$RUNS/$2-s$1
  local cmd=(scripts/train_tiny.py data/science/train.txt --validation-text data/science/validation.txt
    --steps 1000 --eval-interval 250 --gradient-accumulation-steps 4 --device cpu
    --seed "$1" --attention-backend "$2" --output-dir "$out")
  "$PY" "${cmd[@]}" > "$out.log" 2>&1
  "$PY" scripts/record_training_run.py "$out" --protocol science-v1 --code-commit "$COMMIT" \
    --training-command "python ${cmd[*]}" --note "SDPA matched experiment (v1 corpus)"
}
mkdir -p "$RUNS"
train 42 mha; train 42 sdpa
train 43 sdpa; train 43 mha
train 44 mha; train 44 sdpa
for device in cpu mps; do
  "$PY" scripts/evaluate_runs.py "$RUNS"/{mha,sdpa}-s4{2,3,4} --protocol science-v1 --device "$device" \
    --split validation=data/science/validation.txt \
    --split darwin_validation=data/science/darwin-1859-validation.txt \
    --split einstein_validation=data/science/einstein-1924-validation.txt \
    --name "sdpa_v1_eval_$device"
done
