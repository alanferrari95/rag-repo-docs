"""Retrievers: given a question, return the top-k chunks with a score.

DenseRetriever is naive RAG's retrieval: embed the question with the same
model as the chunks and take the k nearest vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk
from .embeddings import Embedder
from .index import VectorIndex


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # higher is better; the scale depends on the retriever


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> list[Hit]: ...


class DenseRetriever:
    name = "dense"

    def __init__(self, index: VectorIndex, embedder: Embedder):
        if embedder.name != index.embedder_name:
            raise ValueError(f"index built with {index.embedder_name}, query embedder is {embedder.name}: "
                             "a question and its chunks must use the same model (rebuild the index)")
        self.index = index
        self.embedder = embedder

    def retrieve(self, question: str, k: int) -> list[Hit]:
        return [Hit(c, s) for c, s in self.index.search(self.embedder.embed_query(question), k)]
