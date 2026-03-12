"""
Off-Policy Evaluation for DPO agent: WIS and FQE.

Uses the SAME methodology as scripts/run_ope.py — see that file for full
documentation of WIS and FQE procedures.

Usage:
    python -m scripts.run_ope_dpo
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.dpo_agent import DPOAgent
from rl.conservative_ql_agent import ReplayBuffer
from rl.q_network import QNetwork


# ---------------------------------------------------------------------------
# Constants  (identical to run_ope.py)
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
# WIS  (identical logic to run_ope.py:compute_wis)
# ---------------------------------------------------------------------------

def compute_wis(buffer, eval_action_prob_fn, gamma=1.0):
    n = len(buffer)
    if n == 0:
        return {"wis_estimate": 0.0, "effective_sample_size": 0.0}

    ratios = []
    rewards = []
    for i in range(n):
        state = buffer._states[i]
        action = buffer._actions[i]
        reward = buffer._rewards[i]

        # Evaluation policy probability
        pi_eval = eval_action_prob_fn(state, action)
        pi_eval = max(pi_eval, 1e-10)

        # Behavior policy probability (uniform approximation)
        # SAME as run_ope.py line 171: "uniform over actions for behavior"
        pi_behav = 1.0 / NUM_ACTIONS
        pi_behav = max(pi_behav, 1e-10)

        ratio = pi_eval / pi_behav
        ratio = np.clip(ratio, WIS_CLIP_MIN, WIS_CLIP_MAX)

        ratios.append(ratio)
        rewards.append(reward)

    ratios = np.array(ratios)
    rewards = np.array(rewards)

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
# FQE  (identical logic to run_ope.py:run_fqe)
# ---------------------------------------------------------------------------

def run_fqe(buffer, eval_action_fn, gamma=1.0,
            num_epochs=FQE_EPOCHS, batch_size=FQE_BATCH_SIZE, lr=FQE_LR):
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

        with torch.no_grad():
            next_actions = []
            for ns in next_states:
                a = eval_action_fn(ns.numpy())
                next_actions.append(a)
            next_actions_t = torch.tensor(next_actions, dtype=torch.long)

            q_next_all = target_q_fqe(next_states)
            q_next_pi = q_next_all.gather(
                1, next_actions_t.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + gamma * (1 - dones) * q_next_pi

        q_all = q_fqe(states)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_sa, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_fqe.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            target_q_fqe.load_state_dict(q_fqe.state_dict())

        losses.append(loss.item())

    # Compute mean Q over full dataset
    all_states = torch.tensor(np.array(buffer._states), dtype=torch.float32)
    all_actions = torch.tensor(buffer._actions, dtype=torch.long)

    with torch.no_grad():
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

def main():
    print("=" * 70)
    print("  Off-Policy Evaluation — DPO")
    print("=" * 70)

    start = time.time()

    # Load buffer
    print(f"\n  Loading buffer from {BUFFER_PATH}...")
    with open(BUFFER_PATH, "rb") as f:
        buffer = pickle.load(f)
    print(f"  Buffer size: {len(buffer)} transitions")

    # Load DPO agent
    print("  Loading DPO agent from runs/dpo_2k/checkpoint.pt...")
    dpo_agent = DPOAgent.load("runs/dpo_2k/checkpoint.pt")

    # -- DPO WIS --
    print("\n  [DPO] Running WIS...")

    def dpo_prob_fn(state, action):
        """Get action probability from DPO policy via softmax(logits).
        Same pattern as IQL in run_ope.py (softmax over policy_network)."""
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32)
            )
            logits = dpo_agent.policy_network(state_t)
            probs = F.softmax(logits, dim=0).numpy()
        if action < len(probs):
            return float(probs[action])
        return 0.0

    wis_dpo = compute_wis(buffer, dpo_prob_fn, GAMMA)
    print(f"    WIS estimate: {wis_dpo['wis_estimate']:.4f}, "
          f"ESS: {wis_dpo['effective_sample_size']:.1f}")

    # -- DPO FQE --
    print("  [DPO] Running FQE...")

    def dpo_action_fn(state):
        """Get greedy action from DPO agent (same as CQL/IQL pattern)."""
        return dpo_agent.select_action(state)

    fqe_dpo = run_fqe(buffer, dpo_action_fn, GAMMA)
    print(f"    FQE mean Q: {fqe_dpo['fqe_mean_q']:.4f}, "
          f"final loss: {fqe_dpo['fqe_final_loss']:.4f}")

    dpo_results = {**wis_dpo, **fqe_dpo}

    elapsed = time.time() - start

    # -- Merge into existing results --
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["dpo"] = dpo_results

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  DPO OPE results saved to: {OUTPUT_PATH}")
    print(f"  Elapsed: {elapsed:.1f}s")

    print("\n  --- DPO OPE Summary ---")
    print(f"  WIS estimate:    {dpo_results['wis_estimate']:.6f}")
    print(f"  WIS ESS:         {dpo_results['effective_sample_size']:.1f}")
    print(f"  FQE mean Q-hat:  {dpo_results['fqe_mean_q']:.6f}")
    print(f"  FQE std Q-hat:   {dpo_results['fqe_std_q']:.6f}")
    print(f"  FQE final loss:  {dpo_results['fqe_final_loss']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
