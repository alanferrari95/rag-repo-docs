"""Retrievers: given a question, return the top-k chunks with a score.

- DenseRetriever: naive RAG's retrieval. Embed the question with the same
  model as the chunks and take the k nearest vectors. Good at paraphrase.
- BM25Retriever: keyword search, words in common weighted by how rare they
  are. Good at exact identifiers (`rollback.sh`, EXTRACT_BACKEND, PSI).
- HybridRetriever: runs both and fuses the two rankings with reciprocal rank
  fusion (RRF), which only uses ranks, so the two score scales never mix.
- RerankRetriever: takes many candidates from any retriever and lets a
  cross-encoder, which reads question and chunk together, pick the best k.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk
from .embeddings import Embedder
from .index import VectorIndex

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"
TOKEN = re.compile(r"[a-z0-9]+")


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


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the same text that is embedded (heading path + chunk)."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.embed_text)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_length = sum(self.lengths) / len(chunks)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, question: str) -> list[float]:
        terms = [t for t in set(tokenize(question)) if t in self.idf]
        scores = []
        for tf, length in zip(self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            scores.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf))
        return scores

    def retrieve(self, question: str, k: int) -> list[Hit]:
        scores = self.score(question)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Hit(self.chunks[i], scores[i]) for i in top]


class HybridRetriever:
    """Reciprocal rank fusion: score(chunk) = sum over retrievers of 1 / (rrf_k + rank)."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], candidates: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.retrieve(question, max(k, self.candidates)), 1):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks[cid] = hit.chunk
        top = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [Hit(chunks[cid], fused[cid]) for cid in top]


class CrossEncoder(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class FastEmbedCrossEncoder:
    def __init__(self, model: str = DEFAULT_RERANKER):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model)

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(question, texts)]


class RerankRetriever:
    """Over-fetch `candidates` chunks quickly, then keep the k the cross-encoder scores highest."""

    def __init__(self, base: Retriever, cross_encoder: CrossEncoder, candidates: int = 30):
        self.base = base
        self.cross_encoder = cross_encoder
        self.candidates = candidates
        self.name = f"{base.name}+rerank"

    def retrieve(self, question: str, k: int) -> list[Hit]:
        hits = self.base.retrieve(question, max(k, self.candidates))
        scores = self.cross_encoder.score(question, [h.chunk.embed_text for h in hits])
        ranked = sorted(zip(scores, hits), key=lambda x: -x[0])[:k]
        return [Hit(h.chunk, s) for s, h in ranked]
