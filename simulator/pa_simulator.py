"""
Public interface for the PA simulator.
Used to generate synthetic PA requests, retrieve policy chunks, and make oracle decisions.
"""

from __future__ import annotations

import os
from typing import List, Optional, Set

import numpy as np

from simulator.oracle import Oracle
from simulator.request_generator import PARequestGenerator
from simulator.retriever import SentenceTransformerRetriever
from simulator.types import PARequest, PolicyChunk

DEFAULT_CORPUS = os.path.join(os.path.dirname(
    __file__), "..", "data", "cms_corpus.json")
DEFAULT_TEMPLATES = os.path.join(os.path.dirname(
    __file__), "..", "data", "templates.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class PASimulator:
    def __init__(self, corpus_path: str = DEFAULT_CORPUS,
                 templates_path: str = DEFAULT_TEMPLATES,
                 model_name: str = DEFAULT_MODEL,
                 seed: int = 42):
        self._generator = PARequestGenerator(templates_path, seed)
        self._retriever = SentenceTransformerRetriever(corpus_path, model_name)
        self._oracle = Oracle(templates_path)

    def generate_request(self, procedure_code: Optional[str] = None) -> PARequest:
        return self._generator.generate(procedure_code)

    def get_corpus(self) -> List[PolicyChunk]:
        return self._retriever.get_corpus()

    def get_chunk(self, chunk_idx: int) -> PolicyChunk:
        return self._retriever.get_chunk(chunk_idx)

    def get_top_k_candidates(self, query_embedding: np.ndarray,
                             k: int = 10,
                             exclude: Optional[Set[int]] = None) -> List[int]:
        return self._retriever.get_top_k(query_embedding, k, exclude)

    def oracle_decision(self, request: PARequest,
                        retrieved_chunks: List[PolicyChunk]) -> str:
        return self._oracle.decide(request, retrieved_chunks)

    def encode(self, text: str) -> np.ndarray:
        return self._retriever.encode(text)

    @property
    def embedding_dim(self) -> int:
        return self._retriever.embedding_dim


if __name__ == "__main__":
    sim = PASimulator()
    request = sim.generate_request(procedure_code="72148")
    print(request)
    candidates = sim.get_top_k_candidates(
        sim.encode(request.procedure_code), k=10)
    print(candidates)
    chunk = sim.get_chunk(candidates[0])
    print(chunk)
    decision = sim.oracle_decision(request, [chunk])
    print(decision)
