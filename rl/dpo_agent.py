"""
Direct Preference Optimization (DPO) agent for the policy retrieval MDP.

Supports two training modes:

1. **Transition-level DPO** (default, ``mode='transition'``):
   For each state in the dataset, constructs per-step preferences by
   comparing transitions from higher-return episodes ("winners") against
   transitions from lower-return episodes ("losers").  This is analogous
   to token-level DPO in language models and preserves per-step gradient
   signal.

2. **Trajectory-level DPO** (``mode='trajectory'``):
   Compares full episode log-probabilities (length-normalised).  This is
   the original DPO formulation adapted for trajectories.

Transition-level DPO overcomes a key limitation of trajectory-level DPO
in sequential MDPs: CQL/IQL use per-transition Bellman updates that can
extrapolate beyond the behavioral data distribution (e.g., learning to
retrieve 20 chunks at eval time from data with 3-8 step episodes).
Trajectory-level DPO cannot extrapolate because it only ranks complete
behavioral trajectories.  Transition-level DPO learns per-step action
quality, enabling the same kind of generalisation.

DPO loss (Rafailov et al., 2023):
    L = -E[ log sigma( beta * (log pi(a_w|s) - log pi_ref(a_w|s)
                              - log pi(a_l|s) + log pi_ref(a_l|s)) ) ]

Reference: Rafailov et al. 2023 — "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model."
DPO stretch plan: R52 in main-doc.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.conservative_ql_agent import ReplayBuffer
from rl.iql_agent import PolicyNetwork


# ---------------------------------------------------------------------------
# Preference dataset (trajectory-level)
# ---------------------------------------------------------------------------

class PreferenceDataset:
    """Dataset of (winner, loser) trajectory pairs for DPO training.

    Constructed from a ReplayBuffer by reconstructing episodes (using the
    ``dones`` flag to detect episode boundaries) and pairing them by
    return.  Each pair consists of two trajectories where the winner has
    strictly higher episodic return than the loser.

    Parameters
    ----------
    margin : float
        Minimum return difference to form a valid preference pair.
        Pairs with |R_w - R_l| < margin are discarded as ambiguous.
        Default 0.5.
    max_pairs : int
        Maximum number of pairs to retain (random subsample if exceeded).
        Default 50_000.
    """

    def __init__(
        self,
        margin: float = 0.5,
        max_pairs: int = 50_000,
    ) -> None:
        self.margin = margin
        self.max_pairs = max_pairs

        # Each trajectory is a dict with keys:
        #   states: Tensor (T, state_dim)
        #   actions: Tensor (T,)
        #   return_: float
        self._winners: List[Dict[str, object]] = []
        self._losers: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._winners)

    @staticmethod
    def _reconstruct_episodes(
        buffer: ReplayBuffer,
    ) -> List[Dict[str, object]]:
        """Reconstruct episodes from a flat ReplayBuffer."""
        episodes: List[Dict[str, object]] = []
        ep_states, ep_actions, ep_rewards = [], [], []

        for idx in range(len(buffer)):
            ep_states.append(buffer._states[idx])
            ep_actions.append(buffer._actions[idx])
            ep_rewards.append(buffer._rewards[idx])

            if buffer._dones[idx]:
                ep_return = sum(ep_rewards)
                episodes.append({
                    "states": torch.tensor(
                        np.array(ep_states), dtype=torch.float32,
                    ),
                    "actions": torch.tensor(ep_actions, dtype=torch.long),
                    "rewards": ep_rewards[:],
                    "return_": ep_return,
                })
                ep_states, ep_actions, ep_rewards = [], [], []

        # Handle case where buffer doesn't end with done=True
        if ep_states:
            ep_return = sum(ep_rewards)
            episodes.append({
                "states": torch.tensor(
                    np.array(ep_states), dtype=torch.float32,
                ),
                "actions": torch.tensor(ep_actions, dtype=torch.long),
                "rewards": ep_rewards[:],
                "return_": ep_return,
            })

        return episodes

    @classmethod
    def from_buffer(
        cls,
        buffer: ReplayBuffer,
        margin: float = 0.5,
        max_pairs: int = 50_000,
        seed: int = 42,
    ) -> "PreferenceDataset":
        """Construct preference pairs from a flat ReplayBuffer.

        Episodes are reconstructed by scanning the ``dones`` flags.
        All unique (i, j) episode pairs where return_i > return_j + margin
        are enumerated, then randomly subsampled to ``max_pairs``.
        """
        rng = random.Random(seed)
        episodes = cls._reconstruct_episodes(buffer)

        # Sort episodes by return for efficient pair generation
        episodes.sort(key=lambda e: e["return_"])
        pairs: List[Tuple[int, int]] = []  # (winner_idx, loser_idx)

        for i in range(len(episodes)):
            for j in range(i):
                if episodes[i]["return_"] - episodes[j]["return_"] >= margin:
                    pairs.append((i, j))

        # Subsample if too many pairs
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)

        dataset = cls(margin=margin, max_pairs=max_pairs)
        for w_idx, l_idx in pairs:
            dataset._winners.append(episodes[w_idx])
            dataset._losers.append(episodes[l_idx])

        return dataset

    def sample(self, batch_size: int) -> Dict[str, List[Dict[str, object]]]:
        """Sample a batch of preference pairs."""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return {
            "winners": [self._winners[i] for i in indices],
            "losers": [self._losers[i] for i in indices],
        }


# ---------------------------------------------------------------------------
# Transition-level preference dataset
# ---------------------------------------------------------------------------

class TransitionPreferenceDataset:
    """Per-step preference pairs for transition-level DPO.

    For each step index t, groups transitions from winner episodes
    (correct decision, return > 0) and loser episodes (incorrect decision,
    return <= 0).  A preference pair is (s_w, a_w, s_l, a_l) where the
    winner transition comes from an episode with higher return.

    This is analogous to token-level DPO: instead of comparing full
    sequences, we compare individual actions in context.  The key
    advantage is that it learns per-step action quality, enabling
    generalisation beyond the behavioral trajectory lengths.
    """

    def __init__(self) -> None:
        # Flat lists of preference pairs
        self._w_states: List[np.ndarray] = []
        self._w_actions: List[int] = []
        self._l_states: List[np.ndarray] = []
        self._l_actions: List[int] = []

    def __len__(self) -> int:
        return len(self._w_states)

    @classmethod
    def from_buffer(
        cls,
        buffer: ReplayBuffer,
        margin: float = 0.5,
        max_pairs: int = 100_000,
        seed: int = 42,
    ) -> "TransitionPreferenceDataset":
        """Construct transition-level preferences from a ReplayBuffer.

        Strategy: separate episodes into winners (return > 0) and losers
        (return <= 0).  For each step index t (0, 1, 2, ...), pair
        transitions from winners with transitions from losers at the same
        step.  This ensures state contexts are roughly comparable (both
        are at the same depth of the retrieval process).

        Additionally, within winners group episodes by return magnitude
        to create preferences between "good" and "great" trajectories,
        providing finer gradient signal.
        """
        rng = random.Random(seed)
        episodes = PreferenceDataset._reconstruct_episodes(buffer)

        # Separate by outcome
        winners = [e for e in episodes if e["return_"] > 0]
        losers = [e for e in episodes if e["return_"] <= 0]

        dataset = cls()

        # --- Strategy 1: Winner vs Loser at same step index ---
        # Group transitions by step index
        max_len = max(len(e["actions"]) for e in episodes)

        for t in range(max_len):
            w_at_t = [(e["states"][t].numpy(), int(e["actions"][t]))
                      for e in winners if len(e["actions"]) > t]
            l_at_t = [(e["states"][t].numpy(), int(e["actions"][t]))
                      for e in losers if len(e["actions"]) > t]

            if not w_at_t or not l_at_t:
                continue

            # Create cross-product pairs (subsample if too many)
            pairs = []
            for ws, wa in w_at_t:
                for ls, la in l_at_t:
                    if wa != la:  # Only if actions differ
                        pairs.append((ws, wa, ls, la))

            if len(pairs) > max_pairs // max_len:
                pairs = rng.sample(pairs, max_pairs // max_len)

            for ws, wa, ls, la in pairs:
                dataset._w_states.append(ws)
                dataset._w_actions.append(wa)
                dataset._l_states.append(ls)
                dataset._l_actions.append(la)

        # --- Strategy 2: High-return vs low-return winners ---
        if len(winners) > 10:
            winners_sorted = sorted(winners, key=lambda e: e["return_"])
            top_quarter = winners_sorted[len(winners_sorted) * 3 // 4:]
            bottom_quarter = winners_sorted[:len(winners_sorted) // 4]

            if top_quarter and bottom_quarter:
                for t in range(max_len):
                    top_at_t = [(e["states"][t].numpy(), int(e["actions"][t]))
                                for e in top_quarter if len(e["actions"]) > t]
                    bot_at_t = [(e["states"][t].numpy(), int(e["actions"][t]))
                                for e in bottom_quarter if len(e["actions"]) > t]

                    if not top_at_t or not bot_at_t:
                        continue

                    pairs = []
                    for ws, wa in top_at_t:
                        for ls, la in bot_at_t:
                            if wa != la:
                                pairs.append((ws, wa, ls, la))

                    if len(pairs) > max_pairs // (2 * max_len):
                        pairs = rng.sample(pairs, max_pairs // (2 * max_len))

                    for ws, wa, ls, la in pairs:
                        dataset._w_states.append(ws)
                        dataset._w_actions.append(wa)
                        dataset._l_states.append(ls)
                        dataset._l_actions.append(la)

        # Final shuffle and cap
        if len(dataset) > max_pairs:
            indices = rng.sample(range(len(dataset)), max_pairs)
            dataset._w_states = [dataset._w_states[i] for i in indices]
            dataset._w_actions = [dataset._w_actions[i] for i in indices]
            dataset._l_states = [dataset._l_states[i] for i in indices]
            dataset._l_actions = [dataset._l_actions[i] for i in indices]

        return dataset

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transition-level preference pairs.

        Returns dict with keys: w_states, w_actions, l_states, l_actions.
        """
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return {
            "w_states": torch.tensor(
                np.array([self._w_states[i] for i in indices]),
                dtype=torch.float32,
            ),
            "w_actions": torch.tensor(
                [self._w_actions[i] for i in indices], dtype=torch.long,
            ),
            "l_states": torch.tensor(
                np.array([self._l_states[i] for i in indices]),
                dtype=torch.float32,
            ),
            "l_actions": torch.tensor(
                [self._l_actions[i] for i in indices], dtype=torch.long,
            ),
        }


# ---------------------------------------------------------------------------
# DPO Agent
# ---------------------------------------------------------------------------

class DPOAgent:
    """Direct Preference Optimization agent for offline RL.

    Supports both transition-level and trajectory-level DPO training.
    Maintains a policy network (learned) and a frozen reference policy.

    Parameters
    ----------
    state_dim : int
        Observation dimensionality (default 768, from StateEncoder).
    num_actions : int
        Number of discrete actions (default 11 = top_k + stop).
    hidden_dim : int
        Hidden layer width (default 256).
    lr : float
        Adam learning rate (default 1e-4).
    beta : float
        DPO temperature (default 0.5). Controls deviation from the
        reference policy.  Lower beta = more conservative.
    grad_clip : float
        Max gradient norm for clipping (default 1.0).
    preference_margin : float
        Minimum return gap to form a preference pair (default 0.5).
    label_smoothing : float
        Label smoothing for DPO loss (default 0.0).  Values > 0 make the
        loss more robust to noisy preferences.
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_actions: int = 11,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        beta: float = 0.5,
        grad_clip: float = 1.0,
        preference_margin: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.beta = beta
        self.grad_clip = grad_clip
        self.preference_margin = preference_margin
        self.label_smoothing = label_smoothing

        # Learned policy
        self.policy_network = PolicyNetwork(state_dim, num_actions, hidden_dim)

        # Frozen reference policy (deep copy; weights set after BC warmup)
        self.ref_policy_network = PolicyNetwork(
            state_dim, num_actions, hidden_dim,
        )
        # Freeze reference
        for p in self.ref_policy_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr,
        )

        self._hyperparams = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "beta": beta,
            "grad_clip": grad_clip,
            "preference_margin": preference_margin,
            "label_smoothing": label_smoothing,
        }

    # ------------------------------------------------------------------
    # Reference policy initialization
    # ------------------------------------------------------------------

    def init_from_bc(self, buffer: ReplayBuffer, bc_epochs: int = 200,
                     batch_size: int = 256) -> Dict[str, List[float]]:
        """Warm-start both policy and reference via behavioral cloning.

        This ensures the reference policy is a reasonable approximation
        of the behavior policy, providing a meaningful KL anchor for DPO.
        After warmup, the reference weights are frozen and the learned
        policy continues to be updated by the DPO loss.

        Parameters
        ----------
        buffer : ReplayBuffer
            Offline dataset for BC pre-training.
        bc_epochs : int
            Number of BC warmup epochs (default 200).
        batch_size : int
            Batch size for BC (default 256).

        Returns
        -------
        dict with key "bc_loss_history" -> list of per-epoch floats.
        """
        metrics: Dict[str, List[float]] = {"bc_loss_history": []}

        self.policy_network.train()
        for epoch in range(1, bc_epochs + 1):
            batch = buffer.sample(batch_size)
            states = batch["states"]
            actions = batch["actions"]

            logits = self.policy_network(states)
            loss = F.cross_entropy(logits, actions)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), self.grad_clip,
            )
            self.optimizer.step()
            metrics["bc_loss_history"].append(loss.item())

        # Freeze reference as a copy of the warmed-up policy
        self.ref_policy_network.load_state_dict(
            self.policy_network.state_dict(),
        )
        for p in self.ref_policy_network.parameters():
            p.requires_grad = False

        return metrics

    # ------------------------------------------------------------------
    # Trajectory log-probability computation (for trajectory-level DPO)
    # ------------------------------------------------------------------

    @staticmethod
    def _trajectory_log_prob(
        network: PolicyNetwork,
        states: torch.Tensor,
        actions: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute (optionally length-normalised) log-prob of a trajectory.

        Parameters
        ----------
        network : PolicyNetwork
        states : torch.Tensor, shape (T, state_dim)
        actions : torch.Tensor, shape (T,)
        normalize : bool
            If True, return mean (per-step average) instead of sum.

        Returns
        -------
        torch.Tensor (scalar)
        """
        logits = network(states)                       # (T, num_actions)
        log_probs = F.log_softmax(logits, dim=-1)      # (T, num_actions)
        action_log_probs = log_probs.gather(
            1, actions.unsqueeze(1),
        ).squeeze(1)                                    # (T,)
        if normalize:
            return action_log_probs.mean()
        return action_log_probs.sum()

    # ------------------------------------------------------------------
    # Transition-level DPO training
    # ------------------------------------------------------------------

    def train_transitions(
        self,
        dataset: TransitionPreferenceDataset,
        num_epochs: int = 500,
        batch_size: int = 256,
        log_every: int = 50,
        writer: Optional[object] = None,
    ) -> Dict[str, List[float]]:
        """Train using transition-level DPO.

        For each epoch:
        1. Sample a batch of (s_w, a_w, s_l, a_l) preference pairs
        2. Compute per-step DPO loss
        3. Backpropagate and update policy network

        Parameters
        ----------
        dataset : TransitionPreferenceDataset
        num_epochs, batch_size, log_every, writer : see train()

        Returns
        -------
        dict with metrics: dpo_loss_history, reward_margin_history,
            accuracy_history.
        """
        metrics: Dict[str, List[float]] = {
            "dpo_loss_history": [],
            "reward_margin_history": [],
            "accuracy_history": [],
        }

        self.policy_network.train()
        self.ref_policy_network.eval()

        for epoch in range(1, num_epochs + 1):
            batch = dataset.sample(batch_size)
            loss, margin, acc = self._train_step_transitions(batch)

            metrics["dpo_loss_history"].append(loss)
            metrics["reward_margin_history"].append(margin)
            metrics["accuracy_history"].append(acc)

            if writer is not None:
                writer.add_scalar("loss/dpo", loss, epoch)
                writer.add_scalar("metrics/reward_margin", margin, epoch)
                writer.add_scalar("metrics/pref_accuracy", acc, epoch)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4d}:  dpo_loss={loss:.4f}  "
                    f"margin={margin:.4f}  pref_acc={acc:.2%}"
                )

        return metrics

    def _train_step_transitions(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, float, float]:
        """Single gradient step on transition-level preference pairs.

        DPO loss per pair:
            -log sigma(beta * (log pi(a_w|s_w) - log pi_ref(a_w|s_w)
                              - log pi(a_l|s_l) + log pi_ref(a_l|s_l)))

        With optional label smoothing for robustness to noisy preferences.
        """
        w_states = batch["w_states"]   # (B, state_dim)
        w_actions = batch["w_actions"] # (B,)
        l_states = batch["l_states"]   # (B, state_dim)
        l_actions = batch["l_actions"] # (B,)

        # Policy log-probs for winner actions at winner states
        w_logits = self.policy_network(w_states)          # (B, num_actions)
        w_log_probs = F.log_softmax(w_logits, dim=-1)
        log_pi_w = w_log_probs.gather(
            1, w_actions.unsqueeze(1)).squeeze(1)          # (B,)

        # Policy log-probs for loser actions at loser states
        l_logits = self.policy_network(l_states)
        l_log_probs = F.log_softmax(l_logits, dim=-1)
        log_pi_l = l_log_probs.gather(
            1, l_actions.unsqueeze(1)).squeeze(1)

        # Reference log-probs (no grad)
        with torch.no_grad():
            w_ref_logits = self.ref_policy_network(w_states)
            w_ref_log_probs = F.log_softmax(w_ref_logits, dim=-1)
            log_ref_w = w_ref_log_probs.gather(
                1, w_actions.unsqueeze(1)).squeeze(1)

            l_ref_logits = self.ref_policy_network(l_states)
            l_ref_log_probs = F.log_softmax(l_ref_logits, dim=-1)
            log_ref_l = l_ref_log_probs.gather(
                1, l_actions.unsqueeze(1)).squeeze(1)

        # DPO reward margins
        log_ratio_w = log_pi_w - log_ref_w  # (B,)
        log_ratio_l = log_pi_l - log_ref_l  # (B,)
        reward_margins = self.beta * (log_ratio_w - log_ratio_l)  # (B,)

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            loss = (
                -(1 - self.label_smoothing) * F.logsigmoid(reward_margins)
                - self.label_smoothing * F.logsigmoid(-reward_margins)
            ).mean()
        else:
            loss = -F.logsigmoid(reward_margins).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.grad_clip,
        )
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            pref_acc = (reward_margins > 0).float().mean().item()
            mean_margin = reward_margins.mean().item()

        return loss.item(), mean_margin, pref_acc

    # ------------------------------------------------------------------
    # Trajectory-level DPO training (with length normalisation)
    # ------------------------------------------------------------------

    def train(
        self,
        pref_dataset: PreferenceDataset,
        num_epochs: int = 500,
        batch_size: int = 64,
        log_every: int = 50,
        writer: Optional[object] = None,
    ) -> Dict[str, List[float]]:
        """Train on trajectory-level preference pairs (length-normalised).

        Parameters
        ----------
        pref_dataset : PreferenceDataset
        num_epochs, batch_size, log_every, writer : standard training args

        Returns
        -------
        dict with metrics.
        """
        metrics: Dict[str, List[float]] = {
            "dpo_loss_history": [],
            "reward_margin_history": [],
            "accuracy_history": [],
        }

        self.policy_network.train()
        self.ref_policy_network.eval()

        for epoch in range(1, num_epochs + 1):
            batch = pref_dataset.sample(batch_size)
            loss, reward_margin, acc = self._train_step_trajectories(batch)

            metrics["dpo_loss_history"].append(loss)
            metrics["reward_margin_history"].append(reward_margin)
            metrics["accuracy_history"].append(acc)

            if writer is not None:
                writer.add_scalar("loss/dpo", loss, epoch)
                writer.add_scalar("metrics/reward_margin", reward_margin, epoch)
                writer.add_scalar("metrics/pref_accuracy", acc, epoch)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4d}:  dpo_loss={loss:.4f}  "
                    f"margin={reward_margin:.4f}  pref_acc={acc:.2%}"
                )

        return metrics

    def _train_step_trajectories(
        self,
        batch: Dict[str, List[Dict[str, object]]],
    ) -> Tuple[float, float, float]:
        """Single gradient step on trajectory-level preference pairs.

        Uses length-normalised log-probabilities (per-step average) to
        prevent bias toward shorter or longer trajectories.
        """
        winners = batch["winners"]
        losers = batch["losers"]

        log_ratios_w = []
        log_ratios_l = []

        for w_traj, l_traj in zip(winners, losers):
            w_states = w_traj["states"]
            w_actions = w_traj["actions"]
            l_states = l_traj["states"]
            l_actions = l_traj["actions"]

            # Length-normalised log-probs
            log_pi_w = self._trajectory_log_prob(
                self.policy_network, w_states, w_actions, normalize=True,
            )
            log_pi_l = self._trajectory_log_prob(
                self.policy_network, l_states, l_actions, normalize=True,
            )

            with torch.no_grad():
                log_ref_w = self._trajectory_log_prob(
                    self.ref_policy_network, w_states, w_actions,
                    normalize=True,
                )
                log_ref_l = self._trajectory_log_prob(
                    self.ref_policy_network, l_states, l_actions,
                    normalize=True,
                )

            log_ratios_w.append(log_pi_w - log_ref_w)
            log_ratios_l.append(log_pi_l - log_ref_l)

        log_ratios_w_t = torch.stack(log_ratios_w)
        log_ratios_l_t = torch.stack(log_ratios_l)

        reward_margins = self.beta * (log_ratios_w_t - log_ratios_l_t)

        if self.label_smoothing > 0:
            loss = (
                -(1 - self.label_smoothing) * F.logsigmoid(reward_margins)
                - self.label_smoothing * F.logsigmoid(-reward_margins)
            ).mean()
        else:
            loss = -F.logsigmoid(reward_margins).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.grad_clip,
        )
        self.optimizer.step()

        with torch.no_grad():
            pref_acc = (reward_margins > 0).float().mean().item()
            mean_margin = reward_margins.mean().item()

        return loss.item(), mean_margin, pref_acc

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
        """Save model checkpoint."""
        checkpoint = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "ref_policy_network_state_dict": (
                self.ref_policy_network.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparameters": self._hyperparams,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "DPOAgent":
        """Load agent from a checkpoint file."""
        checkpoint = torch.load(path, weights_only=False)
        hp = checkpoint["hyperparameters"]
        agent = cls(**hp)
        agent.policy_network.load_state_dict(
            checkpoint["policy_network_state_dict"],
        )
        agent.ref_policy_network.load_state_dict(
            checkpoint["ref_policy_network_state_dict"],
        )
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for p in agent.ref_policy_network.parameters():
            p.requires_grad = False
        return agent
