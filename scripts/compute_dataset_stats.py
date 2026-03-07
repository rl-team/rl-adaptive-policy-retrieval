"""
Compute dataset statistics for M31.

Loads offline_buffer_2k.pkl and test_set_200.pkl and produces:
  1. Total transitions in each buffer
  2. Episodes-per-procedure in train + test
  3. Mean +/- std episode length per behavior policy
  4. Mean return per behavior policy
  5. Decision balance % (approve / deny / pend) across all episodes
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baselines.epsilon_greedy import EpsilonGreedyPolicy
from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy
from rl.conservative_ql_agent import ReplayBuffer
from rl.env import PolicyRetrievalEnv
from rl.reward import RewardFunction
from scripts.collect_offline_dataset import run_baseline_episode
from simulator.pa_simulator import PASimulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reconstruct_episodes(buf: ReplayBuffer) -> List[Dict]:
    """Split a flat ReplayBuffer into per-episode dicts using done flags."""
    episodes = []
    ep_rewards: List[float] = []
    ep_length = 0

    for r, done in zip(buf._rewards, buf._dones):
        ep_rewards.append(r)
        ep_length += 1
        if done:
            episodes.append({"length": ep_length, "return": sum(ep_rewards)})
            ep_rewards = []
            ep_length = 0

    return episodes


def build_policies(seed: int) -> List[Tuple[str, object]]:
    return [
        ("FixedK(k=3)",                 FixedKPolicy(k=3)),
        ("FixedK(k=5)",                 FixedKPolicy(k=5)),
        ("Heuristic(thresh=0.8)",       HeuristicPolicy(confidence_threshold=0.8)),
        ("EpsilonGreedy(eps=0.3,base=FixedK-2)",
         EpsilonGreedyPolicy(FixedKPolicy(k=2), epsilon=0.3, stop_prob=0.3, seed=seed)),
        ("EpsilonGreedy(eps=0.3,base=FixedK-4)", EpsilonGreedyPolicy(FixedKPolicy(k=4),
         epsilon=0.3, stop_prob=0.3, seed=seed + 1)),
    ]


def replay_metadata(n_episodes: int, seed: int) -> List[Dict]:
    """Re-run collection (same seed) to harvest per-episode metadata."""
    sim = PASimulator(seed=seed)
    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=10,
        max_steps=20,
        reward_fn=RewardFunction(step_cost=0.1),
        query_encoder=sim.encode,
    )

    policies = build_policies(seed)
    episodes_per_policy = n_episodes // len(policies)

    meta: List[Dict] = []

    for idx, (policy_name, policy) in enumerate(policies):
        n_eps = episodes_per_policy
        if idx == len(policies) - 1:
            n_eps += n_episodes % len(policies)

        for _ in range(n_eps):
            # Capture env info from reset
            obs, info = env.reset()
            procedure_code = info["procedure_code"]
            policy.reset()

            ep_rewards: List[float] = []
            final_decision = "pend"

            while True:
                candidates = env.candidates
                if policy.should_stop(obs, env.retrieved_chunks):
                    action = env.stop_action
                else:
                    chosen = policy.select_action(obs, candidates)
                    if chosen == -1:
                        action = env.stop_action
                    else:
                        try:
                            action = candidates.index(chosen)
                        except ValueError:
                            action = env.stop_action

                next_obs, reward, terminated, _trunc, step_info = env.step(
                    action)
                ep_rewards.append(reward)
                obs = next_obs

                if terminated:
                    final_decision = step_info.get("decision", "pend")
                    break

            meta.append({
                "policy":    policy_name,
                "procedure": procedure_code,
                "length":    len(ep_rewards),
                "return":    sum(ep_rewards),
                "decision":  final_decision,
            })

    return meta


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(
    train_meta: List[Dict],
    test_meta:  List[Dict],
    train_buf:  ReplayBuffer,
    test_buf:   ReplayBuffer,
) -> str:
    lines: List[str] = []
    sep = "---"

    # -----------------------------------------------------------------------
    # 1. Total transitions
    # -----------------------------------------------------------------------
    lines += [
        "## 1. Total Transitions in Buffer",
        "",
        f"| Dataset | Episodes | Transitions |",
        f"|---------|----------|-------------|",
        f"| Train (`offline_buffer_2k.pkl`) | {len(train_meta):,} | {len(train_buf):,} |",
        f"| Test  (`test_set_200.pkl`)      | {len(test_meta):,}  | {len(test_buf):,}  |",
        "",
    ]

    # -----------------------------------------------------------------------
    # 2. Episodes per procedure
    # -----------------------------------------------------------------------
    train_proc: Dict[str, int] = defaultdict(int)
    test_proc:  Dict[str, int] = defaultdict(int)
    for ep in train_meta:
        train_proc[ep["procedure"]] += 1
    for ep in test_meta:
        test_proc[ep["procedure"]] += 1

    all_procs = sorted(set(train_proc) | set(test_proc))

    lines += [
        "## 2. Episodes per Procedure",
        "",
        f"| Procedure | Train episodes | Test episodes |",
        f"|-----------|---------------|--------------|",
    ]
    for proc in all_procs:
        lines.append(
            f"| `{proc}` | {train_proc.get(proc, 0)} | {test_proc.get(proc, 0)} |")
    lines.append("")

    # -----------------------------------------------------------------------
    # 3 & 4. Episode length and return per behavior policy
    # -----------------------------------------------------------------------
    all_meta = [(m, "train") for m in train_meta] + [(m, "test")
                                                     for m in test_meta]

    by_policy: Dict[str, Dict[str, List[float]]
                    ] = defaultdict(lambda: defaultdict(list))
    for ep, split in all_meta:
        by_policy[ep["policy"]]["length"].append(ep["length"])
        by_policy[ep["policy"]]["return"].append(ep["return"])

    lines += [
        "## 3. Mean +/- Std Episode Length per Behavior Policy",
        "",
        "| Policy | Mean Length | Std Length | N episodes (train+test) |",
        "|--------|-------------|------------|------------------------|",
    ]
    for pol in sorted(by_policy):
        lengths = by_policy[pol]["length"]
        lines.append(
            f"| `{pol}` | {np.mean(lengths):.2f} | {np.std(lengths):.2f} | {len(lengths)} |"
        )
    lines.append("")

    lines += [
        "## 4. Mean Return per Behavior Policy",
        "",
        "| Policy | Mean Return | Std Return |",
        "|--------|-------------|------------|",
    ]
    for pol in sorted(by_policy):
        rets = by_policy[pol]["return"]
        lines.append(
            f"| `{pol}` | {np.mean(rets):.4f} | {np.std(rets):.4f} |"
        )
    lines.append("")

    # -----------------------------------------------------------------------
    # 5. Decision balance
    # -----------------------------------------------------------------------
    def decision_counts(meta: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for ep in meta:
            counts[ep["decision"]] += 1
        return counts

    def fmt_counts(counts: Dict[str, int], total: int) -> str:
        return ", ".join(
            f"{k}: {v} ({100*v/total:.1f}%)"
            for k, v in sorted(counts.items())
        )

    train_d = decision_counts(train_meta)
    test_d = decision_counts(test_meta)
    comb_d: Dict[str, int] = defaultdict(int)
    for ep in train_meta + test_meta:
        comb_d[ep["decision"]] += 1

    all_decisions = sorted(set(train_d) | set(test_d) | set(comb_d))

    lines += [
        "## 5. Decision Balance",
        "",
        "| Decision | Train count | Train % | Test count | Test % | Combined % |",
        "|----------|-------------|---------|------------|--------|------------|",
    ]
    for d in all_decisions:
        tv, tv_p = train_d.get(d, 0), 100 * train_d.get(d, 0) / len(train_meta)
        ev, ev_p = test_d.get(d, 0),  100 * test_d.get(d, 0) / len(test_meta)
        cv_p = 100 * comb_d.get(d, 0) / (len(train_meta) + len(test_meta))
        lines.append(
            f"| {d} | {tv} | {tv_p:.1f}% | {ev} | {ev_p:.1f}% | {cv_p:.1f}% |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    train_path = Path("data/offline_buffer_2k.pkl")
    test_path = Path("data/test_set_200.pkl")

    print("Loading buffers...")
    with open(train_path, "rb") as f:
        train_buf: ReplayBuffer = pickle.load(f)
    with open(test_path, "rb") as f:
        test_buf: ReplayBuffer = pickle.load(f)

    print(f"  Train buffer: {len(train_buf):,} transitions")
    print(f"  Test  buffer: {len(test_buf):,} transitions")

    print("\nReplaying train collection (seed=42) for per-episode metadata...")
    train_meta = replay_metadata(n_episodes=2000, seed=42)

    print("Replaying test collection (seed=99) for per-episode metadata...")
    test_meta = replay_metadata(n_episodes=200,  seed=99)

    print("\nComputing statistics...")
    stats_md = compute_stats(train_meta, test_meta, train_buf, test_buf)

    header = "\n".join([
        "# Dataset Statistics for M31",
        "",
        "Generated from:",
        "- `data/offline_buffer_2k.pkl` (train, seed=42, 2000 episodes)",
        "- `data/test_set_200.pkl` (test, seed=99, 200 episodes)",
        "",
        "Behavior policies (5 total, equal share of episodes):",
        "- FixedK(k=3)",
        "- FixedK(k=5)",
        "- Heuristic(confidence_threshold=0.8)",
        "- EpsilonGreedy(eps=0.3, base=FixedK-2, stop_prob=0.3)",
        "- EpsilonGreedy(eps=0.3, base=FixedK-4, stop_prob=0.3)",
        "",
        "---",
        "",
    ])

    out_path = Path("data/dataset_stats_m31.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + stats_md)
    print(f"\nStats written to {out_path}")
    print("\n" + "=" * 60)
    print(header + stats_md)


if __name__ == "__main__":
    main()
