"""
evaluation harness: orchestrates the evaluation pipeline per EDD

workflow:
1. load test set (same seed => same episode sequence for all policies)
2. for each policy: reset env, run episodes, record correct + chunks per episode
3. compute metrics (accuracy, mean chunks, cost-adjusted utility) with bootstrap CIs
4. return results per policy (visualizations are separate)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
from evaluation.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Run evaluation over policies
# ---------------------------------------------------------------------------
def run_evaluation(
    policies: List[Tuple[str, Callable[[Any], Dict[str, Any]]]],
    num_episodes: int,
    seed: int,
    env_factory: Callable[[int], Any],
    alpha: float = 0.1,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """run all policies on the same test episode sequence and compute metrics

    each policy is evaluated on a fresh environment created with the given seed,
    so the sequence of PA requests is identical across policies (reproducible,
    fair comparison). uses evaluation.metrics.compute_metrics for bootstrap CIs.

    parameters
    ----------
    policies : list of (name, run_episode_fn)
        run_episode_fn(env) runs one episode and returns a dict with at least:
        "correct" (bool), "steps" (int = number of chunks retrieved).
        Optionally may include "return" (float) for per-episode return tracking.

        Reset contract: each ``run_episode_fn`` **must** call ``env.reset()``
        at the start of every episode.  The harness intentionally does *not*
        call reset on behalf of the policy so that the policy can inspect the
        observation and info dict returned by reset (e.g. to read the
        request_id).  Forgetting to call reset will silently re-use the
        previous episode's terminal state and produce wrong results.
    num_episodes : int
        number of episodes per policy
    seed : int
        random seed for env_factory; ensures same request sequence across
        policies (key correctness property for fair comparison).
    env_factory : callable
        env_factory(seed) returns a Gym-like env with reset() and step();
        same seed must yield the same deterministic episode sequence.
    alpha : float
        cost weight for cost-adjusted utility (accuracy - alpha * mean_chunks)
    n_bootstrap : int
        bootstrap samples for 95% CI

    returns
    -------
    dict
        policy_name -> metrics dict (accuracy, accuracy_ci, mean_chunks, ...)
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name, run_episode in policies:
        env = env_factory(seed)
        correct: List[bool] = []
        chunks_per_episode: List[int] = []
        returns: List[float] = []

        for _ in range(num_episodes):
            outcome = run_episode(env)
            correct.append(outcome["correct"])
            chunks_per_episode.append(outcome["steps"])
            if "return" in outcome:
                returns.append(outcome["return"])

        results[name] = compute_metrics(
            correct,
            chunks_per_episode,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            returns_per_episode=returns if returns else None,
        )

    return results
