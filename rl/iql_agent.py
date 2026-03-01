"""
Implicit Q-Learning (IQL) agent for the policy retrieval MDP.

Trains offline on a fixed dataset of (s, a, r, s', done) transitions.
Unlike CQL, IQL avoids direct OOD action penalisation; instead it
extracts the policy *implicitly* through expectile regression and
advantage-weighted behavioural cloning.

Three networks, three losses (Kostrikov et al., 2021):
    V-loss:  L_V  = E[ L₂^τ( Q_target(s,a) - V(s) ) ]
    Q-loss:  L_Q  = E[ (r + γ(1-d)V(s') - Q(s,a))² ]
    π-loss:  L_π  = -E[ exp(β·A(s,a)) · log π(a|s) ]
             where A(s,a) = Q_target(s,a) - V(s)

Reference: Kostrikov et al. 2021 — "Offline Reinforcement Learning
with Implicit Q-Learning".  EDD Use Case 9.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.q_network import QNetwork
from rl.conservative_ql_agent import ReplayBuffer     # shared, DRY


# ---------------------------------------------------------------------------
# Auxiliary networks
# ---------------------------------------------------------------------------

class VNetwork(nn.Module):
    """State-value network V(s) → scalar.

    Same hidden architecture as QNetwork (768 → 256 → 256) but outputs a
    single value instead of per-action Q-values.
    """

    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s) as shape (batch,) or scalar."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Stochastic policy network π(a|s).

    Outputs *logits* (pre-softmax) for each discrete action.  Softmax is
    applied externally so that we can use ``F.cross_entropy`` (which
    expects raw logits) for the advantage-weighted policy loss.
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_actions: int = 11,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape (batch, num_actions) or (num_actions,)."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# IQL Agent
# ---------------------------------------------------------------------------

class IQLAgent:
    """Implicit Q-Learning agent for offline RL.

    Maintains five network copies and trains with three separate losses
    on a fixed replay buffer.

    Networks:
        q_network       – trained Q(s,a) via TD loss using V-targets
        target_q_network – lag-copy of q_network for advantage computation
        v_network       – trained V(s) via asymmetric expectile regression
        policy_network  – trained π(a|s) via advantage-weighted BC

    Parameters
    ----------
    state_dim : int
        Observation dimensionality (default 768).
    num_actions : int
        Number of discrete actions (default 11 = top_k + stop).
    hidden_dim : int
        Hidden layer width for all networks (default 256).
    lr : float
        Adam learning rate for all three optimizers (default 3e-4).
    gamma : float
        Discount factor (default 1.0, undiscounted).
    tau : float
        Expectile for asymmetric V-loss (default 0.7).
        τ > 0.5 biases towards the upper expectile of Q → extracts a
        better-than-average policy.
    beta : float
        Inverse temperature for advantage weighting in the policy loss
        (default 3.0).  Higher β → more greedy extraction.
    target_update_freq : int
        Hard-update target Q-network every N epochs (default 10).
    grad_clip : float
        Max gradient norm for clipping (default 1.0).
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_actions: int = 11,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 1.0,
        tau: float = 0.7,
        beta: float = 3.0,
        target_update_freq: int = 10,
        grad_clip: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # Q-networks (main + lag target for stable advantages)
        self.q_network = QNetwork(state_dim, num_actions, hidden_dim)
        self.target_q_network = self.q_network.copy()

        # V-network
        self.v_network = VNetwork(state_dim, hidden_dim)

        # Policy network
        self.policy_network = PolicyNetwork(state_dim, num_actions, hidden_dim)

        # One Adam optimizer per network (updated sequentially per batch)
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=lr,
        )
        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(), lr=lr,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr,
        )

        # Store hyperparams for checkpointing
        self._hyperparams = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "gamma": gamma,
            "tau": tau,
            "beta": beta,
            "target_update_freq": target_update_freq,
            "grad_clip": grad_clip,
        }

    # ------------------------------------------------------------------
    # Expectile loss helper
    # ------------------------------------------------------------------

    @staticmethod
    def _expectile_loss(
        diff: torch.Tensor, tau: float,
    ) -> torch.Tensor:
        """Asymmetric squared loss L₂^τ(u) = |τ - 1(u<0)| · u².

        When τ > 0.5 more weight is placed on *positive* residuals,
        biasing V(s) towards the upper tail of Q(s,a) — which is exactly
        the value of the *better* actions available in the dataset.
        """
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

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

        For each epoch the three networks are updated sequentially:
            1. V-network  (expectile regression on target Q)
            2. Q-network  (TD loss with V-targets)
            3. Policy      (advantage-weighted BC)

        Parameters
        ----------
        buffer : ReplayBuffer
            Offline dataset of transitions (shared with CQL).
        num_epochs : int
            Training epochs (default 100).
        batch_size : int
            Transitions per batch (default 256).
        log_every : int
            Print progress every N epochs.
        writer : SummaryWriter or None
            Optional Tensorboard writer.

        Returns
        -------
        dict
            Training metrics per epoch.
        """
        metrics: Dict[str, List[float]] = {
            "v_loss_history": [],
            "q_loss_history": [],
            "policy_loss_history": [],
            "q_values_mean_history": [],
        }

        for epoch in range(1, num_epochs + 1):
            batch = buffer.sample(batch_size)
            v_loss, q_loss, pi_loss, q_mean = self._train_step(batch)

            metrics["v_loss_history"].append(v_loss)
            metrics["q_loss_history"].append(q_loss)
            metrics["policy_loss_history"].append(pi_loss)
            metrics["q_values_mean_history"].append(q_mean)

            # Tensorboard logging
            if writer is not None:
                writer.add_scalar("loss/v", v_loss, epoch)
                writer.add_scalar("loss/q_td", q_loss, epoch)
                writer.add_scalar("loss/policy", pi_loss, epoch)
                writer.add_scalar("q_values/mean", q_mean, epoch)

            # Hard-update target Q-network
            if epoch % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(
                    self.q_network.state_dict(),
                )

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4d}:  v_loss={v_loss:.4f}  "
                    f"q_loss={q_loss:.4f}  pi_loss={pi_loss:.4f}  "
                    f"q_mean={q_mean:.4f}"
                )

        return metrics

    def _train_step(
        self, batch: Dict[str, torch.Tensor],
    ) -> tuple[float, float, float, float]:
        """Single gradient step on all three networks.

        Returns (v_loss, q_loss, policy_loss, q_values_mean) as floats.
        """
        states = batch["states"]            # (B, state_dim)
        actions = batch["actions"]          # (B,)
        rewards = batch["rewards"]          # (B,)
        next_states = batch["next_states"]  # (B, state_dim)
        dones = batch["dones"]              # (B,)

        # ── 1. V-network update (expectile regression) ──────────────
        with torch.no_grad():
            target_q_all = self.target_q_network(states)         # (B, A)
            target_q_sa = target_q_all.gather(
                1, actions.unsqueeze(1),
            ).squeeze(1)                                         # (B,)

        v_pred = self.v_network(states)                          # (B,)
        v_loss = self._expectile_loss(target_q_sa - v_pred, self.tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(
            self.v_network.parameters(), self.grad_clip,
        )
        self.v_optimizer.step()

        # ── 2. Q-network update (TD with V-targets) ────────────────
        with torch.no_grad():
            v_next = self.v_network(next_states)                 # (B,)
            q_targets = rewards + self.gamma * (1 - dones) * v_next

        q_all = self.q_network(states)                           # (B, A)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        q_loss = F.mse_loss(q_sa, q_targets)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.grad_clip,
        )
        self.q_optimizer.step()

        # ── 3. Policy-network update (advantage-weighted BC) ────────
        with torch.no_grad():
            # Recompute advantages using *updated* Q but frozen V
            q_adv = self.q_network(states).gather(
                1, actions.unsqueeze(1),
            ).squeeze(1)
            v_adv = self.v_network(states)
            advantages = q_adv - v_adv                           # (B,)

            # Clamp exponent to avoid overflow
            weights = torch.exp(self.beta * advantages)
            weights = torch.clamp(weights, max=100.0)

        logits = self.policy_network(states)                     # (B, A)
        pi_loss = F.cross_entropy(logits, actions, reduction="none")
        pi_loss = (weights * pi_loss).mean()

        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.grad_clip,
        )
        self.policy_optimizer.step()

        return (
            v_loss.item(),
            q_loss.item(),
            pi_loss.item(),
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
        """Select the most probable action under the learned policy.

        Parameters
        ----------
        state : np.ndarray
            Observation vector of shape (state_dim,).
        valid_actions : list of int or None
            If provided, only consider these actions (mask invalid ones).

        Returns
        -------
        int
            Selected action index.
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            logits = self.policy_network(state_t)  # (num_actions,)

        if valid_actions is not None:
            mask = torch.full_like(logits, float("-inf"))
            for a in valid_actions:
                mask[a] = logits[a]
            logits = mask

        return int(logits.argmax().item())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Saves all network weights, optimizer states, and hyperparameters
        so training can be resumed or the model loaded for evaluation.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "v_network_state_dict": self.v_network.state_dict(),
            "policy_network_state_dict": self.policy_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "v_optimizer_state_dict": self.v_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "hyperparameters": self._hyperparams,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "IQLAgent":
        """Load agent from a checkpoint file.

        Returns a fully initialised agent with restored weights.
        """
        checkpoint = torch.load(path, weights_only=False)
        hp = checkpoint["hyperparameters"]
        agent = cls(**hp)
        agent.q_network.load_state_dict(
            checkpoint["q_network_state_dict"],
        )
        agent.target_q_network.load_state_dict(
            checkpoint["target_q_network_state_dict"],
        )
        agent.v_network.load_state_dict(
            checkpoint["v_network_state_dict"],
        )
        agent.policy_network.load_state_dict(
            checkpoint["policy_network_state_dict"],
        )
        agent.q_optimizer.load_state_dict(
            checkpoint["q_optimizer_state_dict"],
        )
        agent.v_optimizer.load_state_dict(
            checkpoint["v_optimizer_state_dict"],
        )
        agent.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer_state_dict"],
        )
        return agent
