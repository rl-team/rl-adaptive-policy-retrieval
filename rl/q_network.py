"""
Q-Network for the policy retrieval MDP.

3-layer MLP that maps states to Q-values for each action. Used by the
Conservative Q-Learning agent as both the main network (trained) and the
target network (updated periodically for stability).

Architecture (EDD Use Case 4, steps 6-8):
    state (768) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(11)

No output activation: Q-values can be negative (e.g. when the agent
retrieves many chunks and still makes an incorrect decision).

Reference: EDD 5.2, Use Case 4.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """MLP value function approximator for the policy retrieval MDP.

    Maps a state vector to Q-values for each discrete action:
        Q(s, a) = estimated return from state s taking action a.

    The Conservative Q-Learning agent maintains two instances:
        - **main**: updated every training step via gradient descent
        - **target**: a lagged copy updated every N epochs for stable
          Bellman targets (EDD Use Case 4, steps 9-11)

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector (default 768, matching the
        base StateEncoder from rl/features.py).
    num_actions : int
        Number of discrete actions (default 11 = top_k + 1 stop action).
    hidden_dim : int
        Width of each hidden layer (default 256, per EDD spec).
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
        """Compute Q-values for all actions given a state (or batch).

        Parameters
        ----------
        state : torch.Tensor
            Shape (state_dim,) for a single state or (batch, state_dim)
            for a batch.

        Returns
        -------
        torch.Tensor
            Shape (num_actions,) or (batch, num_actions).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # no activation — Q-values can be negative

    def copy(self) -> "QNetwork":
        """Create a deep copy for use as a target network.

        The target network has identical weights but independent
        parameters, so gradient updates to the main network do not
        affect the target (EDD Use Case 4, steps 9-11).
        """
        return copy.deepcopy(self)
