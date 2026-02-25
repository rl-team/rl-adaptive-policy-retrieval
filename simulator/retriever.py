"""Sentence-transformer based retriever for policy chunks."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from simulator.types import PolicyChunk

DEFAULT_CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "cms_corpus.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerRetriever:
    def __init__(self, corpus_path: str = DEFAULT_CORPUS,
                 model_name: str = DEFAULT_MODEL):
        self._model = SentenceTransformer(model_name)
        self._corpus: List[PolicyChunk] = []
        self._embeddings: np.ndarray = np.empty(0)
        self._load_corpus(corpus_path)

    def _load_corpus(self, corpus_path: str):
        with open(corpus_path) as f:
            raw = json.load(f)

        texts = [c["text"] for c in raw]
        embeddings = self._model.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False)
        self._embeddings = np.array(embeddings, dtype=np.float32)

        self._corpus = []
        for i, c in enumerate(raw):
            self._corpus.append(PolicyChunk(
                chunk_id=c["chunk_id"],
                policy_id=c["policy_id"],
                text=c["text"],
                embedding=self._embeddings[i],
                section_type=c["section_type"],
                procedure_code=c.get("procedure_code", ""),
                metadata={
                    "index": i,
                },
            ))

    def encode(self, text: str) -> np.ndarray:
        emb = self._model.encode(text, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

    def get_top_k(self, query_embedding: np.ndarray, k: int = 10,
                  exclude: Optional[Set[int]] = None) -> List[int]:
        if exclude is None:
            exclude = set()

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)  # 1e-8 is to avoid division by zero
        sims = self._embeddings @ query_norm  # dot product of embeddings and query norm

        for idx in exclude:
            sims[idx] = -np.inf  # set excluded indices to -inf

        top_indices = np.argsort(sims)[::-1][:k]  # sort by descending similarity
        return [int(i) for i in top_indices if sims[i] > -np.inf]

    def get_corpus(self) -> List[PolicyChunk]:
        return list(self._corpus)

    def get_chunk(self, idx: int) -> PolicyChunk:
        if idx < 0 or idx >= len(self._corpus):
            raise IndexError(f"Chunk index {idx} out of range [0, {len(self._corpus)})")
        return self._corpus[idx]

    @property
    def embedding_dim(self) -> int:
        return self._embeddings.shape[1] if len(self._embeddings) > 0 else 0
