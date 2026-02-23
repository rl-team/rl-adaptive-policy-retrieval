"""
PolicyRetrievalEnv -- Gym environment for the policy retrieval MDP.

Wraps a PA simulator (mock or real) as a standard Gym environment so that
any RL agent or baseline policy can interact with it through reset()/step().

The agent's goal is to retrieve the minimal set of policy chunks needed to
make a correct prior authorization decision, balancing retrieval cost against
decision accuracy.

Reference: EDD 5.2 (PolicyRetrievalEnv), Use Cases 2-3.
"""

from __future__ import annotations

import numpy as np

import gym
from gym import spaces

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rl.features import StateEncoder, EMBEDDING_DIM
from rl.reward import RewardFunction


class PolicyRetrievalEnv(gym.Env):
    """Gym environment for the prior authorization policy retrieval MDP.

    At the start of each episode, a new PA request is generated and the agent
    receives an initial observation (the request embedding). On each step the
    agent either selects a chunk from the top-K candidate list (actions 0..K-1)
    or stops retrieval (action K). Retrieval incurs a per-step cost; stopping
    triggers an oracle decision whose correctness determines the terminal
    reward.

    Parameters
    ----------
    simulator : object
        A PA simulator instance exposing the following methods:
        - generate_request(procedure_code=None) -> request
        - get_corpus() -> list of chunks
        - get_chunk(idx) -> chunk
        - get_top_k_candidates(query_emb, k, exclude) -> list of int
        - oracle_decision(request, chunks) -> str
    top_k : int
        Number of retrieval candidates presented to the agent each step.
    max_steps : int
        Maximum retrieval steps before the episode is forcibly terminated.
    state_encoder : StateEncoder or None
        Pluggable state encoder (EDD Decision 8). Defaults to the base
        768-dim encoder (Option 1). Pass a subclass for Options 2/3.
    reward_fn : RewardFunction or None
        Pluggable reward function (EDD 5.2). Defaults to step cost 0.1
        and +/-1.0 terminal correctness. Pass a subclass for alternatives.
    query_encoder : callable or None
        Injected function that maps text -> np.ndarray embedding.
        Pass ``sim.encode`` when using the real PASimulator (sentence-
        transformer). When ``None``, the env falls back to a deterministic
        hash-based fake embedding for the MockPASimulator.
    """

    # Required by gym.Env. Lists supported rendering modes; we have none
    # since the environment has no visual output.
    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: Any,
        top_k: int = 10,
        max_steps: int = 20,
        state_encoder: Optional[StateEncoder] = None,
        reward_fn: Optional[RewardFunction] = None,
        query_encoder: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        super().__init__()

        self._sim = simulator
        self._top_k = top_k
        self._max_steps = max_steps   # safety bound; corpus has 20 chunks in the mock
        self._encoder = state_encoder or StateEncoder()
        self._reward_fn = reward_fn or RewardFunction()  # default: 0.1 step cost
        self._query_encoder = query_encoder

        # Gym spaces -- observation dim is determined by the encoder,
        # so switching to a larger state (Decision 8) auto-updates this.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._encoder.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(top_k + 1)

        # Precompute the corpus for ground-truth evaluation
        self._corpus = self._sim.get_corpus()

        # Per-episode state (initialized in reset)
        self._request = None               # current PA request
        self._ground_truth: str = ""       # oracle decision with all chunks
        self._retrieved_indices: Set[int] = set()
        self._retrieved_chunks: list = []
        self._candidates: List[int] = []   # current top-K candidate indices
        self._query_embedding: Optional[np.ndarray] = None
        self._steps_taken: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def candidates(self) -> List[int]:
        """Current top-K candidate corpus indices (read-only)."""
        return list(self._candidates)

    @property
    def retrieved_chunks(self) -> list:
        """Chunks retrieved so far in this episode (read-only)."""
        return list(self._retrieved_chunks)

    @property
    def stop_action(self) -> int:
        """The action index that triggers a stop (== top_k)."""
        return self._top_k

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode with a fresh PA request.

        Returns
        -------
        observation : np.ndarray
            Initial 768-dim state vector.
        info : dict
            Episode metadata (request_id, procedure_code, ground_truth).
        """
        super().reset(seed=seed)

        # Generate a new PA request
        procedure_code = (options or {}).get("procedure_code", None)
        self._request = self._sim.generate_request(
            procedure_code=procedure_code,
        )

        # Compute ground truth: oracle decision with ALL chunks, per EDD
        # Use Case 3 ("Ground Truth" section). Cached for terminal reward.
        self._ground_truth = self._sim.oracle_decision(
            self._request, self._corpus,
        )

        # Reset episode state
        self._retrieved_indices = set()
        self._retrieved_chunks = []
        self._steps_taken = 0
        self._done = False

        # Build a query embedding from the request.
        # Uses the injected query_encoder (real system) or a fallback
        # deterministic hash (mock system).
        self._query_embedding = self._build_query_embedding(self._request)

        # Compute initial candidates
        self._candidates = self._sim.get_top_k_candidates(
            self._query_embedding,
            k=self._top_k,
            exclude=self._retrieved_indices,
        )

        obs = self._encoder.encode(
            self._query_embedding, self._retrieved_chunks, self._candidates,
        )
        info = {
            "request_id": self._request.request_id,
            "procedure_code": self._request.procedure_code,
            "ground_truth": self._ground_truth,
        }
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one action in the environment.

        Parameters
        ----------
        action : int
            0..K-1: select the i-th candidate chunk for retrieval.
            K: stop retrieval and trigger oracle decision.

        Returns
        -------
        observation : np.ndarray
            Updated 768-dim state vector.
        reward : float
            -step_cost for retrieval actions, +/-1.0 for terminal decision.
        terminated : bool
            True if episode ended (stop action, max steps, or no candidates).
        truncated : bool
            Always False (no external time limit beyond max_steps).
        info : dict
            Step metadata.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before step()."
            )

        stop_action = self._top_k  # action K = stop

        # -- Stop action or forced stop --
        if action == stop_action or self._steps_taken >= self._max_steps:
            return self._terminate(forced=(action != stop_action))

        # -- Invalid action --
        if action < 0 or action >= len(self._candidates):
            raise ValueError(
                f"Action {action} out of range. "
                f"Valid: 0..{len(self._candidates) - 1} (retrieve) "
                f"or {stop_action} (stop)."
            )

        # -- Retrieval action --
        # Map relative action index to absolute corpus index (EDD Use Case 2,
        # step 9: actual_chunk_idx = candidate_indices[action])
        chunk_idx = self._candidates[action]
        chunk = self._sim.get_chunk(chunk_idx)

        self._retrieved_indices.add(chunk_idx)
        self._retrieved_chunks.append(chunk)
        self._steps_taken += 1

        # Refresh candidate list (exclude already-retrieved chunks)
        self._candidates = self._sim.get_top_k_candidates(
            self._query_embedding,
            k=self._top_k,
            exclude=self._retrieved_indices,
        )

        # If no candidates remain, force termination (EDD Use Case 2, Alt 3)
        if len(self._candidates) == 0:
            return self._terminate(forced=True)

        # Check max_steps after retrieval
        if self._steps_taken >= self._max_steps:
            return self._terminate(forced=True)

        obs = self._encoder.encode(
            self._query_embedding, self._retrieved_chunks, self._candidates,
        )
        info = {
            "chunks_retrieved": len(self._retrieved_chunks),
            "last_chunk_type": chunk.section_type,
        }
        return obs, self._reward_fn.step_reward(), False, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _terminate(
        self, forced: bool = False,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """End the episode, compute terminal reward.

        The oracle evaluates the request using only the chunks the agent
        retrieved.  If that decision matches the ground truth (oracle with
        all chunks), the agent gets +1.0; otherwise -1.0.

        Parameters
        ----------
        forced : bool
            True if termination was forced (max steps or no candidates)
            rather than the agent choosing to stop.
        """
        self._done = True

        # Oracle decision based on what the agent actually retrieved
        if self._retrieved_chunks:
            agent_decision = self._sim.oracle_decision(
                self._request, self._retrieved_chunks,
            )
        else:
            # No chunks retrieved -- oracle defaults to "pend"
            agent_decision = "pend"

        correct = agent_decision == self._ground_truth
        decision_reward = self._reward_fn.terminal_reward(
            agent_decision, self._ground_truth,
        )

        obs = self._encoder.encode(
            self._query_embedding, self._retrieved_chunks, self._candidates,
        )
        info = {
            "decision": agent_decision,
            "ground_truth": self._ground_truth,
            "correct": correct,
            "chunks_retrieved": len(self._retrieved_chunks),
            "forced_stop": forced,
        }
        return obs, decision_reward, True, False, info

    def _build_query_embedding(self, request: Any) -> np.ndarray:
        """Create a query embedding from the PA request.

        Uses the injected query_encoder if available (real system with
        sentence-transformer). Otherwise falls back to a deterministic
        768-dim vector seeded from the request fields (mock system).
        """
        if self._query_encoder is not None:
            return self._query_encoder(request.to_text())

        # Fallback for MockPASimulator: Deterministic 768-dim fake embedding
        # Hash request fields to get a deterministic seed. Modulo 2**31
        # keeps the value within numpy's seed range (non-negative 32-bit int).
        seed_str = (
            f"{request.procedure_code}|"
            f"{'|'.join(sorted(request.diagnosis_codes))}|"
            f"{request.patient_age}|"
            f"{'|'.join(sorted(request.prior_treatments))}"
        )
        seed_val = hash(seed_str) % (2**31)
        rng = np.random.default_rng(seed_val)

        emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        # L2-normalize to unit vector so cosine similarity is meaningful.
        # The 1e-8 epsilon prevents division by zero on degenerate vectors.
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
