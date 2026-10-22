#!/usr/bin/env bash
# Corpus protocol science-v2: build the cleaned corpus, train with the v1
# hyperparameters (unchanged; nothing is tuned on v2), select checkpoints on
# validation, then evaluate each selected checkpoint once on the untouched test book.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
# mha is the legacy path (recorded as "auto" in the published v2 records).
BACKENDS=${BACKENDS:-"mha sdpa"}
RUNS=artifacts/runs/v2
COMMIT=$(git rev-parse --short HEAD)
if [ "${SKIP_PREPARE:-0}" != 1 ]; then "$PY" scripts/prepare_science_corpus_v2.py > /dev/null; fi
git diff --quiet -- data/science-v2 || { echo "data/science-v2 differs from the committed corpus" >&2; exit 1; }
mkdir -p "$RUNS"
runs=()
for seed in 42 43 44; do
  for backend in $BACKENDS; do
    out=$RUNS/$backend-s$seed
    # The test split is deliberately not passed to training.
    cmd=(scripts/train_tiny.py data/science-v2/train.txt --validation-text data/science-v2/validation.txt
      --steps 1000 --eval-interval 250 --gradient-accumulation-steps 4 --device cpu
      --seed "$seed" --attention-backend "$backend" --output-dir "$out")
    "$PY" "${cmd[@]}" > "$out.log" 2>&1
    "$PY" scripts/record_training_run.py "$out" --protocol science-v2 --code-commit "$COMMIT" \
      --train-text data/science-v2/train.txt --validation-text data/science-v2/validation.txt \
      --manifest data/science-v2/manifest.json --training-command "python ${cmd[*]}" --note "science-v2 protocol"
    runs+=("$out")
  done
done
# Final evaluation: once per selected checkpoint, after all training is complete.
"$PY" scripts/evaluate_runs.py "${runs[@]}" --protocol science-v2 --device cpu \
  --split validation=data/science-v2/validation.txt \
  --split darwin_validation=data/science-v2/darwin-1859-validation.txt \
  --split einstein_validation=data/science-v2/einstein-1924-validation.txt \
  --split test=data/science-v2/test.txt \
  --name v2_final_cpu
