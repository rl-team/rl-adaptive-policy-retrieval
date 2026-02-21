from abc import ABC, abstractmethod
from typing import List

from simulator.types import PolicyChunk


class BaselinePolicy(ABC):
    @abstractmethod
    def select_action(self, state, candidates: List[int]) -> int:
        pass

    @abstractmethod
    def should_stop(self, state, history: List[PolicyChunk]) -> bool:
        pass

    def action_prob(self, state, action: int, candidates: List[int],
                    history: List[PolicyChunk]) -> float:
        """Return the probability of taking action in state given history."""
        return 1.0

    def reset(self):
        pass
