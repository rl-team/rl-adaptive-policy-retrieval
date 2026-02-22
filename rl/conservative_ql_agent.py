"""
Conservative Q-Learning agent for the policy retrieval MDP.

Trains offline on a fixed dataset of (s, a, r, s', done) transitions
collected by behavior policies (Fixed-K, Heuristic, epsilon-greedy).
The conservative penalty prevents Q-value overestimation for
out-of-distribution actions, which is critical for offline RL where
the agent cannot explore.

Conservative Q-Learning loss (EDD Use Case 4, step 12):
    L = TD_loss + alpha * conservative_penalty

    TD_loss              = MSE(Q(s,a), r + gamma * max_a' Q_target(s',a'))
    conservative_penalty = (logsumexp(Q(s,:)) - Q(s, a_data)).mean()

Reference: EDD 5.2 (ConservativeQLAgent), Use Case 4.
"""

from __future__ import annotations

import random

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.q_network import QNetwork


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Flat buffer of (s, a, r, s', done) transitions for offline RL.

    Stores transitions from behavior policy rollouts. Each transition is
    a tuple of tensors. Uniform random sampling is used for batching
    (EDD Use Case 4, steps 4-5).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store. FIFO eviction when full.
        Default 100_000 (far exceeds milestone dataset size of ~5K).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._next_states: List[np.ndarray] = []
        self._dones: List[bool] = []

    def __len__(self) -> int:
        return len(self._states)

    def add_episode(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> None:
        """Append all transitions from a single episode.

        Parameters
        ----------
        states, actions, rewards, next_states, dones
            Parallel lists of length T (number of transitions in the episode).
        """
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            if len(self._states) >= self._capacity:
                # FIFO eviction
                self._states.pop(0)
                self._actions.pop(0)
                self._rewards.pop(0)
                self._next_states.pop(0)
                self._dones.pop(0)

            self._states.append(s)
            self._actions.append(a)
            self._rewards.append(r)
            self._next_states.append(ns)
            self._dones.append(d)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a uniform random batch of transitions.

        Returns
        -------
        dict with keys: states, actions, rewards, next_states, dones
            Each value is a tensor sized (batch_size, ...).
        """
        indices = random.sample(range(len(self._states)), batch_size)
        return {
            "states": torch.tensor(
                np.array([self._states[i] for i in indices]),
                dtype=torch.float32,
            ),
            "actions": torch.tensor(
                [self._actions[i] for i in indices],
                dtype=torch.long,
            ),
            "rewards": torch.tensor(
                [self._rewards[i] for i in indices],
                dtype=torch.float32,
            ),
            "next_states": torch.tensor(
                np.array([self._next_states[i] for i in indices]),
                dtype=torch.float32,
            ),
            "dones": torch.tensor(
                [self._dones[i] for i in indices],
                dtype=torch.float32,
            ),
        }


# ---------------------------------------------------------------------------
# ConservativeQL Agent
# ---------------------------------------------------------------------------

class ConservativeQLAgent:
    """Conservative Q-Learning agent for offline RL.

    Maintains two Q-networks (main and target) and trains on a fixed
    replay buffer using the Conservative Q-Learning loss function.

    Parameters
    ----------
    state_dim : int
        Observation dimensionality (default 768, from StateEncoder).
    num_actions : int
        Number of discrete actions (default 11 = top_k + stop).
    hidden_dim : int
        Hidden layer width for QNetwork (default 256).
    lr : float
        Adam learning rate (default 3e-4, per EDD step 13-15).
    gamma : float
        Discount factor (default 1.0, undiscounted for short episodes).
    alpha : float
        Conservative Q-Learning penalty coefficient (default 1.0, per EDD step 12).
        Higher alpha -> more conservative (stays closer to data).
    target_update_freq : int
        Hard-update target network every N epochs (default 10).
    grad_clip : float
        Max gradient norm for clipping (default 1.0, per EDD Alt 1).
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_actions: int = 11,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 1.0,
        alpha: float = 1.0,
        target_update_freq: int = 10,
        grad_clip: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # Main and target Q-networks (EDD: "actually two: main and target")
        self.q_network = QNetwork(state_dim, num_actions, hidden_dim)
        self.target_network = self.q_network.copy()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Store hyperparams for checkpointing
        self._hyperparams = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "gamma": gamma,
            "alpha": alpha,
            "target_update_freq": target_update_freq,
            "grad_clip": grad_clip,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        buffer: ReplayBuffer,
        num_epochs: int = 100,
        batch_size: int = 256,
        log_every: int = 10,
        writer: Optional[object] = None,
    ) -> Dict[str, List[float]]:
        """Train offline on the replay buffer.

        For each epoch:
        1. Sample batch of transitions
        2. Compute Q-targets using target network
        3. Compute ConservativeQL loss (TD error + conservative penalty)
        4. Backpropagate and update Q-network
        5. Every target_update_freq epochs, sync target network

        Parameters
        ----------
        buffer : ReplayBuffer
            Offline dataset of transitions.
        num_epochs : int
            Number of training epochs (default 100).
        batch_size : int
            Transitions per batch (default 256).
        log_every : int
            Print progress every N epochs.
        writer : torch.utils.tensorboard.SummaryWriter or None
            Optional Tensorboard writer for metric logging
            (EDD Use Case 4, steps 16-17). Kept as duck-typed object
            to avoid a hard dependency on tensorboard.

        Returns
        -------
        dict
            Training metrics: loss_history, td_loss_history,
            cql_penalty_history, q_values_mean_history.
        """
        metrics: Dict[str, List[float]] = {
            "loss_history": [],
            "td_loss_history": [],
            "cql_penalty_history": [],
            "q_values_mean_history": [],
        }

        for epoch in range(1, num_epochs + 1):
            batch = buffer.sample(batch_size)
            loss, td_loss, cql_penalty, q_mean = self._train_step(batch)

            metrics["loss_history"].append(loss)
            metrics["td_loss_history"].append(td_loss)
            metrics["cql_penalty_history"].append(cql_penalty)
            metrics["q_values_mean_history"].append(q_mean)

            # Tensorboard logging (EDD step 16-17)
            if writer is not None:
                writer.add_scalar("loss/total", loss, epoch)
                writer.add_scalar("loss/td", td_loss, epoch)
                writer.add_scalar("loss/conservative_penalty", cql_penalty, epoch)
                writer.add_scalar("q_values/mean", q_mean, epoch)

            # Hard-update target network (EDD step 18-19)
            if epoch % self.target_update_freq == 0:
                self.target_network.load_state_dict(
                    self.q_network.state_dict(),
                )

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4d}:  loss={loss:.4f}  "
                    f"td={td_loss:.4f}  conservative_penalty={cql_penalty:.4f}  "
                    f"q_mean={q_mean:.4f}"
                )

        return metrics

    def _train_step(
        self, batch: Dict[str, torch.Tensor],
    ) -> tuple[float, float, float, float]:
        """Single gradient step on a batch of transitions.

        Returns (total_loss, td_loss, cql_penalty, q_values_mean)
        as Python floats for logging.
        """
        states = batch["states"]           # (B, state_dim)
        actions = batch["actions"]         # (B,)
        rewards = batch["rewards"]         # (B,)
        next_states = batch["next_states"] # (B, state_dim)
        dones = batch["dones"]             # (B,)

        # --- Q-values for current state-action pairs ---
        q_all = self.q_network(states)                      # (B, num_actions)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # --- Q-targets using target network (EDD steps 9-11) ---
        with torch.no_grad():
            q_next = self.target_network(next_states)        # (B, num_actions)
            q_next_max = q_next.max(dim=1).values            # (B,)
            targets = rewards + self.gamma * (1 - dones) * q_next_max  # (B,)

        # --- TD loss: MSE Bellman error (EDD step 12) ---
        td_loss = F.mse_loss(q_sa, targets)

        # --- Conservative Q-Learning penalty (EDD step 12) ---
        # logsumexp(Q(s,:)) - Q(s, a_data)
        # Pushes down Q-values for out-of-distribution actions
        logsumexp_q = torch.logsumexp(q_all, dim=1)          # (B,)
        cql_penalty = (logsumexp_q - q_sa).mean()

        # --- Total loss ---
        loss = td_loss + self.alpha * cql_penalty

        # --- Backprop with gradient clipping (EDD Alt 1) ---
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.grad_clip,
        )
        self.optimizer.step()

        return (
            loss.item(),
            td_loss.item(),
            cql_penalty.item(),
            q_all.detach().mean().item(),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        """Select the greedy action (highest Q-value).

        Parameters
        ----------
        state : np.ndarray
            Observation vector of shape (state_dim,).
        valid_actions : list of int or None
            If provided, only consider these actions (mask invalid ones
            to -inf). This is needed because the candidate list shrinks
            as chunks are retrieved.

        Returns
        -------
        int
            Selected action index.
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_t)  # (num_actions,)

        if valid_actions is not None:
            # Mask invalid actions to -inf so argmax ignores them
            mask = torch.full_like(q_values, float("-inf"))
            for a in valid_actions:
                mask[a] = q_values[a]
            q_values = mask

        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model checkpoint (EDD step 21-22).

        Saves Q-network weights, optimizer state, and hyperparameters
        so training can be resumed or the model can be loaded for eval.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparameters": self._hyperparams,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "ConservativeQLAgent":
        """Load agent from a checkpoint file.

        Returns a fully initialized agent with restored weights.
        """
        checkpoint = torch.load(path, weights_only=False)
        hp = checkpoint["hyperparameters"]
        agent = cls(**hp)
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.target_network.load_state_dict(
            checkpoint["target_network_state_dict"],
        )
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent
