"""Tests for evaluation.metrics (EDD Metrics Calculator)."""

import numpy as np

from evaluation.metrics import (
    accuracy,
    mean_chunks,
    cost_adjusted_utility,
    precision_at_k,
    bootstrap_ci,
    compute_metrics,
)


def test_accuracy():
    assert accuracy([True, True, False]) == 2 / 3
    assert accuracy([True]) == 1.0
    assert accuracy([]) == 0.0


def test_mean_chunks():
    assert mean_chunks([3, 5, 4]) == 4.0
    assert mean_chunks([]) == 0.0


def test_cost_adjusted_utility():
    correct = [True, True, False]
    chunks = [2, 4, 6]
    u = cost_adjusted_utility(correct, chunks, alpha=0.1)
    assert abs(u - (2 / 3 - 0.4)) < 1e-9


def test_precision_at_k():
    retrieved = [[1, 2, 3], [1, 2]]
    relevant = [[1, 2], [1, 2, 3]]
    p = precision_at_k(retrieved, relevant)
    # ep0: 2/3 relevant; ep1: 2/2 relevant -> mean (2/3 + 1)/2
    assert abs(p - (2 / 3 + 1) / 2) < 1e-9


def test_bootstrap_ci():
    values = np.ones(100) * 0.5
    low, high = bootstrap_ci(values, n_bootstrap=500)
    assert low <= 0.5 <= high


def test_compute_metrics():
    correct = [True] * 80 + [False] * 20
    chunks = [3] * 50 + [5] * 50
    out = compute_metrics(correct, chunks, alpha=0.1, n_bootstrap=100)
    assert out["accuracy"] == 0.8
    assert out["mean_chunks"] == 4.0
    assert out["cost_adjusted_utility"] == 0.8 - 0.1 * 4.0
    assert out["n_episodes"] == 100
    assert isinstance(out["accuracy_ci"], tuple)
    assert len(out["accuracy_ci"]) == 2
