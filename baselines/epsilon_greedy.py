"""Epsilon-greedy baseline: wraps another policy, with probability epsilon
takes a random action for exploration."""

from __future__ import annotations
from typing import List

import numpy as np

from simulator.types import PolicyChunk
from baselines.base import BaselinePolicy


class EpsilonGreedyPolicy(BaselinePolicy):
    def __init__(self, base_policy: BaselinePolicy, epsilon: float = 0.3,
                 stop_prob: float = 0.3, seed: int = 42):
        self.base = base_policy
        self.epsilon = epsilon
        self.stop_prob = stop_prob
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self.base.reset()

    def select_action(self, state, candidates: List[int]) -> int:
        if not candidates:
            return -1
        if self._rng.random() < self.epsilon:
            return candidates[self._rng.integers(len(candidates))]
        return self.base.select_action(state, candidates)

    def should_stop(self, state, history: List[PolicyChunk]) -> bool:
        if self._rng.random() < self.epsilon:
            return self._rng.random() < self.stop_prob
        return self.base.should_stop(state, history)

    def action_prob(self, state, action: int, candidates: List[int],
                    history: List[PolicyChunk]) -> float:
        # Probability of stopping
        base_stops = self.base.should_stop(state, history)
        p_stop = (self.epsilon * self.stop_prob) + ((1 - self.epsilon) * (1.0 if base_stops else 0.0))
        
        if action == -1:
            return p_stop
            
        # Probability of continuing and picking action
        p_continue = 1.0 - p_stop
        if p_continue == 0:
            return 0.0
            
        n_candidates = len(candidates)
        if n_candidates == 0:
            return 0.0
            
        # P(action | continue)
        p_random = 1.0 / n_candidates
        p_base = 1.0 if self.base.select_action(state, candidates) == action else 0.0
        
        p_action_given_continue = (self.epsilon * p_random) + ((1 - self.epsilon) * p_base)
        
        return p_continue * p_action_given_continue
