"""Heuristic baseline: computes confidence from retrieved chunks,
stops when confidence exceeds threshold."""

from __future__ import annotations
from typing import List

from simulator.types import PolicyChunk
from baselines.base import BaselinePolicy


class HeuristicPolicy(BaselinePolicy):
    def __init__(self, confidence_threshold: float = 0.8):
        self.threshold = confidence_threshold

    def select_action(self, state, candidates: List[int]) -> int:
        if not candidates:
            return -1
        return candidates[0]

    def should_stop(self, state, history: List[PolicyChunk]) -> bool:
        return self._compute_confidence(history) >= self.threshold

    def _compute_confidence(self, history: List[PolicyChunk]) -> float:
        if not history:
            return 0.0
        has_coverage = any(c.section_type == "coverage_criteria" for c in history)
        base = 0.7 if has_coverage else 0.3
        return min(base + 0.1 * len(history), 1.0)

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
