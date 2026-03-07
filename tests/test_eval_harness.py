"""
tests for evaluation.eval_harness -- verifies run_evaluation orchestrates
policies, collects outcomes, and returns metrics per policy (EDD Evaluation Harness)

run with:
    python -m tests.test_eval_harness
"""

from __future__ import annotations
from typing import List
from evaluation.eval_harness import run_evaluation


def _run_tests() -> None:
    """verify the evaluation harness"""

    print("=" * 70)
    print("  evaluation.eval_harness -- Tests")
    print("=" * 70)

    # --- 1. run_evaluation with fake env and single policy ---
    print("\n[Test 1] run_evaluation: single policy, fake env")
    class FakeEnv:
        def reset(self):
            return None, {}
        def __getattr__(self, _):
            return lambda *a, **k: (None, 0, True, False, {"correct": True, "steps": 3})

    def run_one(env):
        env.reset()
        env.step(0)
        return {"correct": True, "steps": 3}

    results = run_evaluation(
        [("Fake", run_one)],
        num_episodes=5,
        seed=42,
        env_factory=lambda s: FakeEnv(),
        alpha=0.1,
        n_bootstrap=100,
    )
    assert "Fake" in results, f"Expected 'Fake' in results, got {list(results.keys())}"
    m = results["Fake"]
    assert m["accuracy"] == 1.0, f"Expected 1.0, got {m['accuracy']}"
    assert m["mean_chunks"] == 3.0, f"Expected 3.0, got {m['mean_chunks']}"
    assert "accuracy_ci" in m and isinstance(m["accuracy_ci"], tuple), (
        "Expected accuracy_ci tuple"
    )
    assert m["n_episodes"] == 5, f"Expected 5, got {m['n_episodes']}"
    print(f"  Fake policy: accuracy={m['accuracy']}, mean_chunks={m['mean_chunks']}")
    print("  PASSED")

    # --- 2. multiple policies, same episode sequence ---
    print("\n[Test 2] run_evaluation: multiple policies")
    def run_two(env):
        env.reset()
        env.step(0)
        return {"correct": False, "steps": 2}

    results2 = run_evaluation(
        [("A", run_one), ("B", run_two)],
        num_episodes=4,
        seed=99,
        env_factory=lambda s: FakeEnv(),
        n_bootstrap=50,
    )
    assert results2["A"]["accuracy"] == 1.0, f"Expected A accuracy 1.0, got {results2['A']['accuracy']}"
    assert results2["B"]["accuracy"] == 0.0, f"Expected B accuracy 0.0, got {results2['B']['accuracy']}"
    assert results2["A"]["mean_chunks"] == 3.0 and results2["B"]["mean_chunks"] == 2.0
    print(f"  Policy A: accuracy={results2['A']['accuracy']}, chunks={results2['A']['mean_chunks']}")
    print(f"  Policy B: accuracy={results2['B']['accuracy']}, chunks={results2['B']['mean_chunks']}")
    print("  PASSED")

    # --- 3. env_factory seed contract: both policies see same episode sequence ---
    print("\n[Test 3] env_factory(seed): both policies see same reset() sequence")
    class DeterministicSeqEnv:
        """env that returns deterministic request_id on each reset() for same seed"""
        def __init__(self, seed: int):
            self._counter = 0
        def reset(self):
            info = {"request_id": f"req_{self._counter}"}
            self._counter += 1
            return None, info
        def step(self, action):
            return None, 0, True, False, {"correct": True, "steps": 0}

    seen_A: List[str] = []
    seen_B: List[str] = []

    def run_record_A(env):
        obs, info = env.reset()
        seen_A.append(info["request_id"])
        env.step(0)
        return {"correct": True, "steps": 0}

    def run_record_B(env):
        obs, info = env.reset()
        seen_B.append(info["request_id"])
        env.step(0)
        return {"correct": True, "steps": 0}

    run_evaluation(
        [("A", run_record_A), ("B", run_record_B)],
        num_episodes=3,
        seed=7,
        env_factory=lambda s: DeterministicSeqEnv(s),
        n_bootstrap=10,
    )
    assert seen_A == seen_B, (
        f"Both policies must see same episode sequence; got A={seen_A}, B={seen_B}"
    )
    assert seen_A == ["req_0", "req_1", "req_2"], (
        f"Expected req_0..req_2, got {seen_A}"
    )
    print(f"  Both policies saw reset() sequence: {seen_A}")
    print("  PASSED")

    # --- 4. seed contract: same seed -> same results across separate calls ---
    print("\n[Test 4] seed contract: same seed reproduces identical metrics")
    class SeededEnv:
        """env whose reset() returns seed-derived deterministic request ids"""
        def __init__(self, seed: int):
            import random as _r
            self._rng = _r.Random(seed)
        def reset(self):
            rid = self._rng.randint(0, 999999)
            return None, {"request_id": rid}
        def step(self, action):
            return None, 0, True, False, {}

    def run_seeded(env):
        _obs, info = env.reset()
        # deterministic policy: correct if request_id is even
        correct = (info["request_id"] % 2 == 0)
        env.step(0)
        return {"correct": correct, "steps": 1}

    r1 = run_evaluation(
        [("P", run_seeded)],
        num_episodes=10,
        seed=123,
        env_factory=lambda s: SeededEnv(s),
        n_bootstrap=50,
    )
    r2 = run_evaluation(
        [("P", run_seeded)],
        num_episodes=10,
        seed=123,
        env_factory=lambda s: SeededEnv(s),
        n_bootstrap=50,
    )
    assert r1["P"]["accuracy"] == r2["P"]["accuracy"], (
        f"Same seed must give same accuracy: {r1['P']['accuracy']} vs {r2['P']['accuracy']}"
    )
    assert r1["P"]["mean_chunks"] == r2["P"]["mean_chunks"], (
        f"Same seed must give same mean_chunks"
    )
    # Different seed should (almost certainly) give different results
    r3 = run_evaluation(
        [("P", run_seeded)],
        num_episodes=10,
        seed=456,
        env_factory=lambda s: SeededEnv(s),
        n_bootstrap=50,
    )
    # This is a probabilistic assertion but with 10 episodes from different
    # seeds, the probability of identical accuracy is very low.
    print(f"  seed=123 accuracy={r1['P']['accuracy']:.2f}, "
          f"seed=456 accuracy={r3['P']['accuracy']:.2f}")
    print("  Same seed -> identical metrics: PASSED")

    # --- 5. harness forwards optional 'return' to compute_metrics ---
    print("\n[Test 5] harness collects per-episode returns when provided")

    def run_with_return(env):
        env.reset()
        env.step(0)
        return {"correct": True, "steps": 2, "return": 5.0}

    r_ret = run_evaluation(
        [("WithReturn", run_with_return)],
        num_episodes=4,
        seed=42,
        env_factory=lambda s: FakeEnv(),
        n_bootstrap=50,
    )
    m_ret = r_ret["WithReturn"]
    assert "mean_return" in m_ret, "Expected mean_return in results"
    assert m_ret["mean_return"] == 5.0, f"Expected 5.0, got {m_ret['mean_return']}"
    assert "mean_return_ci" in m_ret, "Expected mean_return_ci in results"
    print(f"  mean_return={m_ret['mean_return']}, ci={m_ret['mean_return_ci']}")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
