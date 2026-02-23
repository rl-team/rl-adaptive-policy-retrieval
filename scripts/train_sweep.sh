#!/usr/bin/env bash
# Hyperparameter sweep for Conservative Q-Learning training (R15).
#
# Runs 3 configurations to explore the alpha/lr tradeoff:
#   1. Baseline:  alpha=1.0, lr=3e-4  (EDD defaults)
#   2. Moderate:  alpha=0.5, lr=1e-3  (less conservative, faster lr)
#   3. Minimal:   alpha=0.1, lr=3e-4  (near-standard DQN)
#
# Each run goes to a separate directory under runs/ for Tensorboard comparison.
# After training, evaluate with:
#   python -m scripts.evaluate_agent --checkpoint runs/<name>/checkpoint.pt
#
# Monitor all runs with:
#   tensorboard --logdir runs/
#
# Reference: EDD Use Case 4 (hyperparameter α controls conservatism).

set -euo pipefail

DATASET="data/offline_buffer.pkl"
EPOCHS=200

echo "============================================================"
echo "  Conservative Q-Learning Hyperparameter Sweep"
echo "  Dataset: ${DATASET}"
echo "  Epochs:  ${EPOCHS}"
echo "============================================================"

# Config 1: EDD baseline defaults
echo ""
echo "--- Config 1: alpha=1.0, lr=3e-4 (EDD defaults) ---"
python -m scripts.train_conservative_ql \
    --dataset "${DATASET}" \
    --epochs "${EPOCHS}" \
    --alpha 1.0 \
    --lr 3e-4 \
    --output runs/sweep_alpha1.0_lr3e-4

# Config 2: User's moderate config
echo ""
echo "--- Config 2: alpha=0.5, lr=1e-3 (moderate) ---"
python -m scripts.train_conservative_ql \
    --dataset "${DATASET}" \
    --epochs "${EPOCHS}" \
    --alpha 0.5 \
    --lr 1e-3 \
    --output runs/sweep_alpha0.5_lr1e-3

# Config 3: Minimal conservatism
echo ""
echo "--- Config 3: alpha=0.1, lr=3e-4 (minimal conservatism) ---"
python -m scripts.train_conservative_ql \
    --dataset "${DATASET}" \
    --epochs "${EPOCHS}" \
    --alpha 0.1 \
    --lr 3e-4 \
    --output runs/sweep_alpha0.1_lr3e-4

echo ""
echo "============================================================"
echo "  Sweep complete. Compare runs in Tensorboard:"
echo "    tensorboard --logdir runs/"
echo ""
echo "  Evaluate best checkpoint:"
echo "    python -m scripts.evaluate_agent --checkpoint runs/<best>/checkpoint.pt"
echo "============================================================"
