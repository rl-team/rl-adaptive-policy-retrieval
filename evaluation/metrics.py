"""
metrics for policy retrieval evaluation using EDD metrics calculator

- Accuracy: num_correct / total_episodes
- Avg Chunks: mean(num_chunks_retrieved)
- Precision@K: retrieval quality (relevant chunks retrieved)
- Cost-Adjusted Utility: accuracy - alpha * avg_chunks

comparison method: mean and 95% confidence interval using bootstrap
resampling (1000 samples)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Core metrics (EDD Metrics Calculator)
# ---------------------------------------------------------------------------
def accuracy(correct: List[bool]) -> float:
    """accuracy: num_correct / total_episodes"""
    if not correct:
        return 0.0
    return float(np.mean(correct))


def mean_chunks(chunks_per_episode: List[int]) -> float:
    """avg chunks: mean(num_chunks_retrieved) per episode"""
    if not chunks_per_episode:
        return 0.0
    return float(np.mean(chunks_per_episode))


def cost_adjusted_utility(
    correct: List[bool],
    chunks_per_episode: List[int],
    alpha: float = 0.1,
) -> float:
    """cost-adjusted utility: accuracy - alpha * avg_chunks (composite metric)"""
    acc = accuracy(correct)
    avg_chunks = mean_chunks(chunks_per_episode)
    return acc - alpha * avg_chunks


def precision_at_k(
    retrieved_per_episode: List[List[int]],
    relevant_per_episode: List[List[int]],
    k: Optional[int] = None,
) -> float:
    """precision@K: fraction of retrieved chunks that are relevant
    if k is None, uses the number of chunks actually retrieved per episode
    parameters
    ----------
    retrieved_per_episode : list of list of int
        retrieved chunk ids for each episode (in order retrieved)
    relevant_per_episode : list of list of int
        ground-truth relevant chunk ids for each episode
    k : int or None
        consider only top-k retrieved; if None, use full retrieval set

    returns
    -------
    float
        mean over episodes of (|retrieved ∩ relevant| / min(k, |retrieved|))
    """
    if len(retrieved_per_episode) != len(relevant_per_episode):
        raise ValueError("retrieved_per_episode and relevant_per_episode must have same length")
    if not retrieved_per_episode:
        return 0.0

    precisions = []
    for retrieved, relevant in zip(retrieved_per_episode, relevant_per_episode):
        rel_set = set(relevant)
        if k is not None:
            retrieved = retrieved[:k]
        if not retrieved:
            precisions.append(0.0)
            continue
        n_relevant_retrieved = sum(1 for c in retrieved if c in rel_set)
        precisions.append(n_relevant_retrieved / len(retrieved))
    return float(np.mean(precisions))


# ---------------------------------------------------------------------------
# Bootstrap 95% CI (1000 samples per EDD)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic_fn: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """95% confidence interval using bootstrap resampling (1000 samples)

    EDD: "For each metric, compute mean and 95% confidence interval
    using bootstrap resampling (1000 samples)."

    parameters
    ----------
    values : np.ndarray
        one-dimensional array of per-episode values (e.g. 0/1 correct, or chunks)
    statistic_fn : str
        one of "mean", "sum"; applied to each bootstrap sample
    n_bootstrap : int
        number of bootstrap samples (default 1000 per EDD)
    confidence : float
        confidence level (default 0.95 for 95% CI)

    returns
    -------
    (lower, upper) : tuple
        confidence interval bounds
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        if statistic_fn == "mean":
            stats.append(float(np.mean(sample)))
        elif statistic_fn == "sum":
            stats.append(float(np.sum(sample)))
        else:
            raise ValueError(f"Unknown statistic_fn: {statistic_fn}")

    low = (1 - confidence) / 2
    high = 1 - low
    lower = float(np.quantile(stats, low))
    upper = float(np.quantile(stats, high))
    return (lower, upper)


# ---------------------------------------------------------------------------
# Aggregate metrics + CIs
# ---------------------------------------------------------------------------

def compute_metrics(
    correct: List[bool],
    chunks_per_episode: List[int],
    alpha: float = 0.1,
    n_bootstrap: int = 1000,
    retrieved_per_episode: Optional[List[List[int]]] = None,
    relevant_per_episode: Optional[List[List[int]]] = None,
    returns_per_episode: Optional[List[float]] = None,
) -> Dict[str, float | Tuple[float, float]]:
    """compute all metrics with bootstrap 95% CIs

    returns a dict with:
    - accuracy, accuracy_ci
    - mean_chunks, mean_chunks_ci
    - cost_adjusted_utility, cost_adjusted_utility_ci
    - n_episodes
    - returns_per_episode (raw list, only if provided)
    - mean_return, mean_return_ci (only if returns_per_episode provided)
    - precision_at_k, precision_at_k_ci (only if relevance data provided)
    """
    n = len(correct)
    if n != len(chunks_per_episode):
        raise ValueError("correct and chunks_per_episode must have same length")

    correct_arr = np.array(correct, dtype=float)
    chunks_arr = np.array(chunks_per_episode, dtype=float)
    cau_arr = correct_arr - alpha * chunks_arr

    acc = accuracy(correct)
    acc_ci = bootstrap_ci(correct_arr, "mean", n_bootstrap)
    mean_ch = mean_chunks(chunks_per_episode)
    mean_ch_ci = bootstrap_ci(chunks_arr, "mean", n_bootstrap)
    cau = cost_adjusted_utility(correct, chunks_per_episode, alpha)
    cau_ci = bootstrap_ci(cau_arr, "mean", n_bootstrap)

    out = {
        "accuracy": acc,
        "accuracy_ci": acc_ci,
        "mean_chunks": mean_ch,
        "mean_chunks_ci": mean_ch_ci,
        "cost_adjusted_utility": cau,
        "cost_adjusted_utility_ci": cau_ci,
        "n_episodes": n,
    }

    if returns_per_episode is not None:
        if len(returns_per_episode) != n:
            raise ValueError(
                "returns_per_episode must have same length as correct"
            )
        returns_arr = np.array(returns_per_episode, dtype=float)
        out["returns_per_episode"] = list(returns_per_episode)
        out["mean_return"] = float(np.mean(returns_arr))
        out["mean_return_ci"] = bootstrap_ci(returns_arr, "mean", n_bootstrap)

    if retrieved_per_episode is not None and relevant_per_episode is not None:
        prec = precision_at_k(retrieved_per_episode, relevant_per_episode)
        # per-episode precision values for bootstrap CI
        prec_per_ep = []
        for ret, rel in zip(retrieved_per_episode, relevant_per_episode):
            rel_set = set(rel)
            if not ret:
                prec_per_ep.append(0.0)
            else:
                prec_per_ep.append(sum(1 for c in ret if c in rel_set) / len(ret))
        prec_arr = np.array(prec_per_ep, dtype=float)
        prec_ci = bootstrap_ci(prec_arr, "mean", n_bootstrap)
        out["precision_at_k"] = prec
        out["precision_at_k_ci"] = prec_ci

    return out
