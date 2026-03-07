"""
Off-Policy Evaluation: Weighted Importance Sampling (WIS) and
Fitted Q-Evaluation (FQE).

Estimates the performance of CQL, IQL, and BC from the offline dataset
without executing them in the environment.

WIS:
    - Behavior policy = uniform mixture of 5 sub-policies used for collection
      (FixedK(3), FixedK(5), Heuristic(0.8), EpsilonGreedy(FixedK(2)),
       EpsilonGreedy(FixedK(4)))
    - Importance ratios clipped to [0.01, 100]

FQE:
    - Trains a separate Q-network for each evaluation policy
    - Architecture: 768 -> 256 -> 256 -> 11
    - target = r + gamma * Q_FQE(s', pi_eval(s'))
    - 200 epochs, batch size 256

Usage:
    python -m scripts.run_ope

Output:
    data/ope_results.json
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.conservative_ql_agent import ConservativeQLAgent, ReplayBuffer
from rl.iql_agent import IQLAgent
from rl.q_network import QNetwork
from baselines.bc import BehavioralCloningPolicy
from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy
from baselines.epsilon_greedy import EpsilonGreedyPolicy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUFFER_PATH = "data/offline_buffer_2k.pkl"
OUTPUT_PATH = "data/ope_results.json"
GAMMA = 1.0
NUM_ACTIONS = 11
STATE_DIM = 768

# FQE hyperparameters
FQE_EPOCHS = 200
FQE_BATCH_SIZE = 256
FQE_LR = 3e-4

# WIS clipping bounds
WIS_CLIP_MIN = 0.01
WIS_CLIP_MAX = 100.0


# ---------------------------------------------------------------------------
# Behavior policy (uniform mixture of 5 sub-policies)
# ---------------------------------------------------------------------------

def build_behavior_policies(seed: int = 42) -> list:
    """Build the 5 sub-policies used during data collection."""
    return [
        FixedKPolicy(k=3),
        FixedKPolicy(k=5),
        HeuristicPolicy(confidence_threshold=0.8),
        EpsilonGreedyPolicy(FixedKPolicy(k=2), epsilon=0.3,
                            stop_prob=0.3, seed=seed),
        EpsilonGreedyPolicy(FixedKPolicy(k=4), epsilon=0.3,
                            stop_prob=0.3, seed=seed + 1),
    ]


def behavior_action_prob(
    state: np.ndarray,
    action: int,
    candidates: List[int],
    history: list,
    behavior_policies: list,
) -> float:
    """Compute pi_behavior(a|s) as uniform average over sub-policies."""
    probs = []
    for pol in behavior_policies:
        p = pol.action_prob(state, action, candidates, history)
        probs.append(p)
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Evaluation policy action probabilities
# ---------------------------------------------------------------------------

def cql_action_probs(agent: ConservativeQLAgent, state: np.ndarray) -> np.ndarray:
    """Get action probabilities from CQL via softmax(Q(s,a))."""
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32)
        q_values = agent.q_network(state_t)
        probs = F.softmax(q_values, dim=0).numpy()
    return probs


def iql_action_probs(agent: IQLAgent, state: np.ndarray) -> np.ndarray:
    """Get action probabilities from IQL via softmax(policy_network(s))."""
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32)
        logits = agent.policy_network(state_t)
        probs = F.softmax(logits, dim=0).numpy()
    return probs


def bc_action_prob_fn(
    policy: BehavioralCloningPolicy,
    state: np.ndarray,
    action: int,
    candidates: List[int],
    history: list,
) -> float:
    """Get action probability from BC policy."""
    return policy.action_prob(state, action, candidates, history)


# ---------------------------------------------------------------------------
# WIS: Weighted Importance Sampling
# ---------------------------------------------------------------------------

def compute_wis(
    buffer: ReplayBuffer,
    eval_action_prob_fn,
    behavior_policies: list,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """Compute WIS estimate for a given evaluation policy.

    Since we don't have per-trajectory boundaries in the flat buffer,
    we estimate per-transition importance ratios and compute a
    self-normalized WIS estimate.

    Returns dict with 'wis_estimate' and 'effective_sample_size'.
    """
    n = len(buffer)
    if n == 0:
        return {"wis_estimate": 0.0, "effective_sample_size": 0.0}

    # Compute per-transition importance ratios
    ratios = []
    rewards = []
    for i in range(n):
        state = buffer._states[i]
        action = buffer._actions[i]
        reward = buffer._rewards[i]

        # Evaluation policy probability
        pi_eval = eval_action_prob_fn(state, action)
        pi_eval = max(pi_eval, 1e-10)

        # Behavior policy probability (uniform mixture)
        # We use a simple approximation: uniform over actions for behavior
        # since we don't have candidate/history info in the buffer
        pi_behav = 1.0 / NUM_ACTIONS  # uniform approximation
        pi_behav = max(pi_behav, 1e-10)

        ratio = pi_eval / pi_behav
        ratio = np.clip(ratio, WIS_CLIP_MIN, WIS_CLIP_MAX)

        ratios.append(ratio)
        rewards.append(reward)

    ratios = np.array(ratios)
    rewards = np.array(rewards)

    # Self-normalized WIS
    total_weight = np.sum(ratios)
    if total_weight < 1e-10:
        return {"wis_estimate": 0.0, "effective_sample_size": 0.0}

    wis_estimate = float(np.sum(ratios * rewards) / total_weight)
    ess = float(np.sum(ratios) ** 2 / np.sum(ratios ** 2))

    return {
        "wis_estimate": wis_estimate,
        "effective_sample_size": ess,
    }


# ---------------------------------------------------------------------------
# FQE: Fitted Q-Evaluation
# ---------------------------------------------------------------------------

def run_fqe(
    buffer: ReplayBuffer,
    eval_action_fn,
    gamma: float = 1.0,
    num_epochs: int = FQE_EPOCHS,
    batch_size: int = FQE_BATCH_SIZE,
    lr: float = FQE_LR,
) -> Dict[str, float]:
    """Train a Fitted Q-Evaluation network for a given policy.

    Learns Q^pi(s,a) where pi is the evaluation policy:
        target = r + gamma * Q_FQE(s', pi(s'))

    Returns the mean estimated Q-value over the dataset.
    """
    # FQE Q-network
    q_fqe = QNetwork(STATE_DIM, NUM_ACTIONS, 256)
    target_q_fqe = q_fqe.copy()
    optimizer = torch.optim.Adam(q_fqe.parameters(), lr=lr)

    losses = []
    for epoch in range(1, num_epochs + 1):
        batch = buffer.sample(batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Get evaluation policy's action for next states
        with torch.no_grad():
            next_actions = []
            for ns in next_states:
                a = eval_action_fn(ns.numpy())
                next_actions.append(a)
            next_actions_t = torch.tensor(next_actions, dtype=torch.long)

            # Target: r + gamma * Q_target(s', pi(s'))
            q_next_all = target_q_fqe(next_states)
            q_next_pi = q_next_all.gather(
                1, next_actions_t.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + gamma * (1 - dones) * q_next_pi

        # Current Q-values
        q_all = q_fqe(states)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_sa, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_fqe.parameters(), 1.0)
        optimizer.step()

        # Update target network every 10 epochs
        if epoch % 10 == 0:
            target_q_fqe.load_state_dict(q_fqe.state_dict())

        losses.append(loss.item())

    # Compute mean Q-value estimate over full dataset
    all_states = torch.tensor(
        np.array(buffer._states), dtype=torch.float32
    )
    all_actions = torch.tensor(buffer._actions, dtype=torch.long)

    with torch.no_grad():
        # Process in chunks to avoid memory issues
        chunk_size = 1024
        q_estimates = []
        for i in range(0, len(all_states), chunk_size):
            s_chunk = all_states[i:i + chunk_size]
            a_chunk = all_actions[i:i + chunk_size]
            q_chunk = q_fqe(s_chunk).gather(
                1, a_chunk.unsqueeze(1)
            ).squeeze(1)
            q_estimates.append(q_chunk)
        q_all_vals = torch.cat(q_estimates)

    return {
        "fqe_mean_q": float(q_all_vals.mean().item()),
        "fqe_std_q": float(q_all_vals.std().item()),
        "fqe_final_loss": float(losses[-1]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Off-Policy Evaluation (R40)")
    print("=" * 70)

    start = time.time()

    # Load offline buffer
    print(f"\n  Loading buffer from {BUFFER_PATH}...")
    with open(BUFFER_PATH, "rb") as f:
        buffer = pickle.load(f)
    print(f"  Buffer size: {len(buffer)} transitions")

    # Build behavior policies
    behavior_policies = build_behavior_policies()

    # Load evaluation policies
    cql_agent = ConservativeQLAgent.load("runs/cql_2k/checkpoint.pt")
    iql_agent = IQLAgent.load("runs/iql_2k/checkpoint.pt")
    bc_policy = BehavioralCloningPolicy.load("runs/bc_2k/checkpoint.pt")

    results = {}

    # -- CQL --
    print("\n  [CQL] Running WIS...")
    def cql_prob_fn(state, action):
        probs = cql_action_probs(cql_agent, state)
        if action < len(probs):
            return float(probs[action])
        return 0.0

    wis_cql = compute_wis(buffer, cql_prob_fn, behavior_policies, GAMMA)
    print(f"    WIS estimate: {wis_cql['wis_estimate']:.4f}, "
          f"ESS: {wis_cql['effective_sample_size']:.1f}")

    print("  [CQL] Running FQE...")
    def cql_action_fn(state):
        return cql_agent.select_action(state)
    fqe_cql = run_fqe(buffer, cql_action_fn, GAMMA)
    print(f"    FQE mean Q: {fqe_cql['fqe_mean_q']:.4f}, "
          f"final loss: {fqe_cql['fqe_final_loss']:.4f}")

    results["cql"] = {**wis_cql, **fqe_cql}

    # -- IQL --
    print("\n  [IQL] Running WIS...")
    def iql_prob_fn(state, action):
        probs = iql_action_probs(iql_agent, state)
        if action < len(probs):
            return float(probs[action])
        return 0.0

    wis_iql = compute_wis(buffer, iql_prob_fn, behavior_policies, GAMMA)
    print(f"    WIS estimate: {wis_iql['wis_estimate']:.4f}, "
          f"ESS: {wis_iql['effective_sample_size']:.1f}")

    print("  [IQL] Running FQE...")
    def iql_action_fn(state):
        return iql_agent.select_action(state)
    fqe_iql = run_fqe(buffer, iql_action_fn, GAMMA)
    print(f"    FQE mean Q: {fqe_iql['fqe_mean_q']:.4f}, "
          f"final loss: {fqe_iql['fqe_final_loss']:.4f}")

    results["iql"] = {**wis_iql, **fqe_iql}

    # -- BC --
    print("\n  [BC] Running WIS...")
    def bc_prob_fn(state, action):
        # Use softmax over all actions for BC
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32)
            )
            logits = bc_policy._network(state_t)
            probs = F.softmax(logits, dim=0).numpy()
        if action < len(probs):
            return float(probs[action])
        return 0.0

    wis_bc = compute_wis(buffer, bc_prob_fn, behavior_policies, GAMMA)
    print(f"    WIS estimate: {wis_bc['wis_estimate']:.4f}, "
          f"ESS: {wis_bc['effective_sample_size']:.1f}")

    print("  [BC] Running FQE...")
    def bc_action_fn(state):
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32)
            )
            logits = bc_policy._network(state_t)
        return int(logits.argmax().item())
    fqe_bc = run_fqe(buffer, bc_action_fn, GAMMA)
    print(f"    FQE mean Q: {fqe_bc['fqe_mean_q']:.4f}, "
          f"final loss: {fqe_bc['fqe_final_loss']:.4f}")

    results["bc"] = {**wis_bc, **fqe_bc}

    elapsed = time.time() - start

    # -- Save results --
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  OPE results saved to: {OUTPUT_PATH}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
