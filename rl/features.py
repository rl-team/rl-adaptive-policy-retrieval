"""
State encoders for the policy retrieval MDP.

Transforms the raw retrieval context (request, retrieved chunks, candidates)
into fixed-size observation vectors for the RL agent. Different encoders
implement the observation strategies from EDD Decision 8.

Reference: EDD 5.2 (StateEncoder), Decision 8 (observation strategies).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768      # sentence-transformer embedding dimensionality

# The EDD (5.2) specifies the base observation as two equal halves:
#   dims   0..383 = request embedding
#   dims 384..767 = mean-pooled chunk embeddings
HALF_DIM = EMBEDDING_DIM // 2


# ---------------------------------------------------------------------------
# State Encoder (pluggable, per EDD Decision 8)
#
# EDD Decision 8 defines three observation strategies that we want to
# ablate over in the final report:
#   Option 1: 768-dim  (request_emb + history_emb)       -- milestone
#   Option 2: 774-dim  (+ 6 compact candidate features)  -- final report
#   Option 3: 4608-dim (+ K full candidate embeddings)   -- final report
#
# By delegating state encoding to a separate object, switching strategies
# is a one-line change at env construction time. New encoders just need
# to implement obs_dim and encode().
# ---------------------------------------------------------------------------

class StateEncoder:
    """Base state encoder: 768-dim observation (EDD Decision 8, Option 1).

    Subclass and override ``obs_dim`` and ``encode()`` to implement Options 2
    or 3 for the final report ablation.
    """

    @property
    def obs_dim(self) -> int:
        """Dimensionality of the observation vector."""
        return EMBEDDING_DIM  # 768

    def encode(
        self,
        query_embedding: np.ndarray,
        retrieved_chunks: list,
        candidates: list,
    ) -> np.ndarray:
        """Encode the current retrieval context into a fixed-size vector.

        Parameters
        ----------
        query_embedding : np.ndarray
            768-dim query vector for the current PA request.
        retrieved_chunks : list
            Chunks retrieved so far (each has a .embedding attribute).
        candidates : list
            Current top-K candidate indices (unused in Option 1, but
            available for Options 2/3 that need candidate information).

        Returns
        -------
        np.ndarray of shape (obs_dim,)
        """
        # Request component: first half of query embedding
        request_emb = query_embedding[:HALF_DIM]

        # Chunks component: mean-pool retrieved chunk embeddings (second half)
        if retrieved_chunks:
            # Use [-HALF_DIM:] to handle both 768-dim fakes (takes last half)
            # and 384-dim real embeddings (takes the whole array)
            chunk_embs = np.stack(
                [c.embedding[-HALF_DIM:] for c in retrieved_chunks],
                axis=0,
            )
            chunks_emb = np.mean(chunk_embs, axis=0)
        else:
            chunks_emb = np.zeros(HALF_DIM, dtype=np.float32)

        return np.concatenate([request_emb, chunks_emb]).astype(np.float32)
