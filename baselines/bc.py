"""
Behavioral Cloning (BC) baseline for the policy retrieval MDP.

Learns a policy by supervised classification on (state, action) pairs
from the offline dataset — no reward signal, no TD bootstrapping.
Serves as the simplest learned baseline: if CQL/IQL cannot outperform
BC, the RL formulation adds no value.

Training objective (cross-entropy):
    L_BC = -E_{(s,a)~D}[ log π(a | s) ]

Reference: EDD Use Case 10 (R27).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.q_network import QNetwork   # reuse architecture: 768→256→256→11
from rl.conservative_ql_agent import ReplayBuffer
from baselines.base import BaselinePolicy
from simulator.types import PolicyChunk


class BehavioralCloningPolicy(BaselinePolicy):
    """Behavioral Cloning baseline: supervised action prediction.

    Internally uses the same 3-layer MLP as QNetwork but interprets the
    output as *action logits* (not Q-values).  At inference time the
    policy selects the most probable action according to softmax(logits).

    Conforms to the ``BaselinePolicy`` ABC, so ``evaluate_agent.py`` can
    treat it like any other baseline (FixedK, Heuristic, etc.).

    Parameters
    ----------
    state_dim : int
        Observation dimensionality (default 768).
    num_actions : int
        Number of discrete actions (default 11 = top_k + stop).
    hidden_dim : int
        Hidden layer width (default 256).
    lr : float
        Adam learning rate (default 1e-3).
    stop_action : int
        The action index that corresponds to "stop retrieval".
        When ``select_action`` picks this action, ``should_stop``
        returns True.  Default equals ``num_actions - 1`` (= 10).
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_actions: int = 11,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        stop_action: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self._stop_action = (
            stop_action if stop_action is not None else num_actions - 1
        )

        # Reuse QNetwork architecture (same dims, different semantics)
        self._network = QNetwork(state_dim, num_actions, hidden_dim)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)

        self._hyperparams = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "stop_action": self._stop_action,
        }

    # ------------------------------------------------------------------
    # Training (supervised)
    # ------------------------------------------------------------------

    def train(
        self,
        buffer: ReplayBuffer,
        num_epochs: int = 100,
        batch_size: int = 256,
        log_every: int = 10,
    ) -> Dict[str, List[float]]:
        """Train via cross-entropy on (state, action) pairs.

        Parameters
        ----------
        buffer : ReplayBuffer
            Same offline dataset used by CQL/IQL (only states + actions
            are used; rewards and next_states are ignored).
        num_epochs : int
            Training epochs (default 100).
        batch_size : int
            Transitions per batch (default 256).
        log_every : int
            Print progress every N epochs.

        Returns
        -------
        dict with key "loss_history" → list of per-epoch loss floats.
        """
        metrics: Dict[str, List[float]] = {"loss_history": []}

        self._network.train()
        for epoch in range(1, num_epochs + 1):
            batch = buffer.sample(batch_size)
            states = batch["states"]       # (B, state_dim)
            actions = batch["actions"]     # (B,)

            logits = self._network(states)  # (B, num_actions)
            loss = F.cross_entropy(logits, actions)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            loss_val = loss.item()
            metrics["loss_history"].append(loss_val)

            if epoch % log_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4d}:  loss={loss_val:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # BaselinePolicy interface
    # ------------------------------------------------------------------

    def select_action(self, state, candidates: List[int]) -> int:
        """Pick the most probable action restricted to ``candidates``.

        Parameters
        ----------
        state : np.ndarray
            Observation vector of shape (state_dim,).
        candidates : list of int
            Valid candidate **corpus indices**.  Internally, the network
            outputs logits over relative action indices (0..K-1), so we
            mask logits to the first ``len(candidates)`` positions and
            return the corresponding corpus index.

        Returns
        -------
        int
            Chosen candidate corpus index, or -1 if no candidates.
        """
        if not candidates:
            return -1

        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32),
            )
            logits = self._network(state_t)  # (num_actions,)

        # Mask to valid relative action indices (0..len(candidates)-1)
        mask = torch.full_like(logits, float("-inf"))
        for i in range(min(len(candidates), self.num_actions - 1)):
            mask[i] = logits[i]

        best_action = int(mask.argmax().item())

        # Map relative action index back to absolute corpus index
        if best_action < len(candidates):
            return candidates[best_action]
        return -1

    def should_stop(self, state, history: List[PolicyChunk]) -> bool:
        """Decide whether to stop by comparing stop vs best-continue.

        Uses the full action set (stop included) to make the comparison:
        if the model's highest-logit action is the stop action, return
        True; otherwise return False.
        """
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32),
            )
            logits = self._network(state_t)

        return int(logits.argmax().item()) == self._stop_action

    def action_prob(
        self,
        state,
        action: int,
        candidates: List[int],
        history: List[PolicyChunk],
    ) -> float:
        """Return π(action | state) for importance-weighting.

        ``action`` is a relative action index (0..K-1 for retrieval, or
        the stop action index).  Computes a softmax over valid action
        indices + stop, then returns the probability of ``action``.
        """
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32),
            )
            logits = self._network(state_t)

        # Build valid action set: relative indices 0..len(candidates)-1 + stop
        n_cands = min(len(candidates), self.num_actions - 1)
        valid = list(range(n_cands)) + [self._stop_action]
        mask = torch.full_like(logits, float("-inf"))
        for a in valid:
            mask[a] = logits[a]

        probs = F.softmax(mask, dim=0)

        if action == -1:
            return probs[self._stop_action].item()
        return probs[action].item()

    def reset(self) -> None:
        """No per-episode state to reset."""
        pass

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save checkpoint (network weights + hyperparameters)."""
        checkpoint = {
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "hyperparameters": self._hyperparams,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "BehavioralCloningPolicy":
        """Load a trained BC policy from a checkpoint file."""
        checkpoint = torch.load(path, weights_only=False)
        hp = checkpoint["hyperparameters"]
        policy = cls(**hp)
        policy._network.load_state_dict(checkpoint["network_state_dict"])
        policy._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return policy
