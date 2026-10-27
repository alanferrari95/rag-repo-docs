"""A vector index that is just a matrix.

With a few hundred chunks, exact search is one matrix-vector product: every
vector is unit length, so the dot product is the cosine similarity. Approximate
indexes (HNSW in pgvector, ScaNN in Vertex AI Vector Search) only pay off at
hundreds of thousands of vectors.

On disk: vectors.npy + chunks.jsonl + meta.json in one directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> VectorIndex:
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        return cls(chunks, vectors, embedder.name)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vector
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]  # the k best, unordered: O(n)
        top = top[np.argsort(-scores[top])]  # then sort only those k
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "n_chunks": len(self.chunks),
                        "dim": int(self.vectors.shape[1])}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embedder"])
