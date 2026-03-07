#!/usr/bin/env bash
# Lambda (step_cost) ablation sweep for CQL.
#
# Collects a separate offline buffer for each lambda value (because the
# reward is baked into the buffer at collection time), then trains CQL
# on each buffer.  Produces 3 checkpoints ready for evaluation.
#
# Reference: Implementation Plan R35.
#
# Usage:
#     bash scripts/train_lambda_sweep.sh
#     bash scripts/train_lambda_sweep.sh 2>&1 | tee logs/lambda_sweep.log

set -euo pipefail

EPISODES=2000
SEED=42
EPOCHS=200
ALPHA=1.0
LR=3e-4

echo "======================================================================"
echo "  R35: Lambda Ablation Sweep (CQL)"
echo "======================================================================"
echo "  Episodes per buffer: ${EPISODES}"
echo "  CQL epochs: ${EPOCHS}, alpha=${ALPHA}, lr=${LR}"
echo ""

for LAMBDA in 0.05 0.1 0.2; do
  BUFFER="data/offline_buffer_2k_lambda${LAMBDA}.pkl"
  OUTDIR="runs/sweep_lambda_${LAMBDA}"

  echo "--- Lambda = ${LAMBDA} ---"

  # Step 1: Collect offline buffer with this step_cost
  python -m scripts.collect_offline_dataset \
    --episodes "${EPISODES}" --seed "${SEED}" \
    --step-cost "${LAMBDA}" \
    --output "${BUFFER}"

  # Step 2: Train CQL on the matching buffer
  python -m scripts.train_conservative_ql \
    --dataset "${BUFFER}" --epochs "${EPOCHS}" \
    --alpha "${ALPHA}" --lr "${LR}" \
    --output "${OUTDIR}"

  echo "  Checkpoint: ${OUTDIR}/checkpoint.pt"
  echo ""
done

echo "======================================================================"
echo "  Lambda sweep complete.  Checkpoints:"
for LAMBDA in 0.05 0.1 0.2; do
  echo "    runs/sweep_lambda_${LAMBDA}/checkpoint.pt"
done
echo "======================================================================"
