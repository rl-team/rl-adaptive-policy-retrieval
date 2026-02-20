"""
Reward functions for the policy retrieval MDP.

Computes per-step and terminal rewards based on retrieval cost and decision
correctness. The base RewardFunction implements the cost-benefit tradeoff
from EDD 5.2:

    step_reward     = -step_cost        (per retrieval action)
    terminal_reward = +1.0 if correct, -1.0 if incorrect

Subclass to implement alternative reward designs for ablation experiments
(e.g. sparse rewards, shaped rewards with partial credit).

Reference: EDD 5.2 (RewardFunction), Use Cases 2-3.
"""

from __future__ import annotations


class RewardFunction:
    """Base reward function: per-step cost + binary terminal correctness.

    EDD 5.2 defines the reward as:
        r_t = -lambda           for each retrieval action
        r_T = +1.0 if correct,  -1.0 if incorrect

    where lambda (step_cost) controls the cost-benefit tradeoff between
    retrieving more chunks and making a faster decision. The EDD recommends
    ablating over {0.05, 0.1, 0.2}.

    Subclass and override ``step_reward()`` and ``terminal_reward()`` to
    experiment with alternative designs, for example:

    - **Sparse**: +1 if correct and chunks <= budget, else 0
    - **Shaped**: +1 correct, +0.5 pend (partial credit), 0 wrong

    Parameters
    ----------
    step_cost : float
        Cost subtracted from the reward for each retrieval action
        (lambda in the EDD). Default 0.1.
    """

    def __init__(self, step_cost: float = 0.1) -> None:
        self._step_cost = step_cost

    @property
    def step_cost(self) -> float:
        """The per-retrieval cost (lambda)."""
        return self._step_cost

    def step_reward(self) -> float:
        """Reward for a single retrieval action.

        Returns -step_cost. Called by the environment on each non-terminal
        step (EDD Use Case 2, step 11).
        """
        return -self._step_cost

    def terminal_reward(self, decision: str, ground_truth: str) -> float:
        """Reward at episode termination based on decision correctness.

        Compares the agent's decision (from the oracle using retrieved chunks)
        to the ground truth (oracle using all chunks). Returns +1.0 for a
        correct match, -1.0 otherwise (EDD Use Case 3, reward computation).

        Parameters
        ----------
        decision : str
            Oracle decision based on the agent's retrieved chunks
            ("approve", "deny", or "pend").
        ground_truth : str
            Oracle decision based on all available chunks.

        Returns
        -------
        float
            +1.0 if decision == ground_truth, -1.0 otherwise.
        """
        return 1.0 if decision == ground_truth else -1.0
