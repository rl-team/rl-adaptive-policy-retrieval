"""Fixed-K baseline: always retrieve exactly K chunks, then stop."""

from __future__ import annotations
from typing import List

from simulator.types import PolicyChunk
from baselines.base import BaselinePolicy


class FixedKPolicy(BaselinePolicy):
    def __init__(self, k: int = 5):
        self.k = k

    def select_action(self, state, candidates: List[int]) -> int:
        if not candidates:
            return -1
        return candidates[0]

    def should_stop(self, state, history: List[PolicyChunk]) -> bool:
        return len(history) >= self.k

    def action_prob(self, state, action: int, candidates: List[int],
                    history: List[PolicyChunk]) -> float:
        stop_prob = 1.0 if self.should_stop(state, history) else 0.0

        if action == -1:
            return stop_prob

        if stop_prob == 1.0:
            return 0.0  # Should have stopped

        # If continuing
        if self.select_action(state, candidates) == action:
            return 1.0
        return 0.0
