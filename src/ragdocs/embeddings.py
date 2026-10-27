"""Turn text into unit-length vectors.

Two embedders behind the same small interface:
- FastEmbedEmbedder: a real sentence-embedding model (BAAI/bge-small-en-v1.5,
  384 dimensions) run locally with ONNX. It downloads ~70 MB once and needs no
  key or GPU.
- HashingEmbedder: hashed bag of words. It has no idea that "rollback" and
  "revert" are related, but it is deterministic and needs no download, which
  is what the offline tests need.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # imported here so tests never load it

        self.name = model
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models expect an instruction prefix on queries; query_embed adds it.
        return _normalize(np.array(list(self._model.query_embed([text]))))[0]


TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 2048):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))


def make_embedder(name: str) -> Embedder:
    if name.startswith("hashing"):
        _, _, dim = name.partition("-")
        return HashingEmbedder(int(dim) if dim else 2048)
    return FastEmbedEmbedder(name)
