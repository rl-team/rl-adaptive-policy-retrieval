"""
Evaluation package for policy retrieval experiments.

Provides metrics (accuracy, avg chunks, cost-adjusted utility, bootstrap CI)
per EDD "Evaluation Components (Hannah's Ownership)".
"""

from evaluation.metrics import (
    compute_metrics,
    accuracy,
    mean_chunks,
    cost_adjusted_utility,
    precision_at_k,
    bootstrap_ci,
)

__all__ = [
    "compute_metrics",
    "accuracy",
    "mean_chunks",
    "cost_adjusted_utility",
    "precision_at_k",
    "bootstrap_ci",
]
