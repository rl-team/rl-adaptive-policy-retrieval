"""
tests for evaluation.eval_harness -- verifies run_evaluation orchestrates
policies, collects outcomes, and returns metrics per policy (EDD Evaluation Harness)

run with:
    python -m tests.test_eval_harness
"""

from __future__ import annotations

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

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
