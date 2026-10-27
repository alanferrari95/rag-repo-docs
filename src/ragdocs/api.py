"""HTTP API.

    uvicorn ragdocs.api:app

POST /ask {"question": "...", "k": 5, "backend": "extractive"} returns the answer,
whether it abstained, and each retrieved chunk with its [n], score and a link
to the source file at the pinned commit. GET /health reports the index.

RAGDOCS_INDEX (default index/) and RAGDOCS_RETRIEVER (dense | hybrid |
hybrid+rerank, default hybrid+rerank) configure it; build the index first
with `python -m ragdocs build`. The extractive answer abstains below the
threshold `python -m ragdocs eval` picked on the dev set (or RAGDOCS_MIN_SCORE).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .corpus import DEFAULT_CORPUS
from .embeddings import Embedder, make_embedder
from .generation import ExtractiveGenerator, LLMGenerator
from .index import VectorIndex
from .llm import BACKENDS, make_llm
from .pipeline import RAG
from .retrieval import (
    BM25Retriever, DenseRetriever, FastEmbedCrossEncoder, HybridRetriever, RerankRetriever, Retriever,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "index"
RESULTS = ROOT / "eval" / "results" / "results.json"


def default_min_score() -> float:
    """Abstention threshold for extractive answers: RAGDOCS_MIN_SCORE, else the one `ragdocs eval` picked on dev."""
    if os.getenv("RAGDOCS_MIN_SCORE"):
        return float(os.environ["RAGDOCS_MIN_SCORE"])
    if RESULTS.exists():
        return float(json.loads(RESULTS.read_text(encoding="utf-8"))["answers"]["threshold_from_dev"])
    return 0.0


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    backend: str = "extractive"


class Source(BaseModel):
    n: int
    chunk_id: str
    heading: str
    score: float
    cited: bool
    url: str | None


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    backend: str
    retriever: str
    sources: list[Source]


def source_urls(corpus: Path = DEFAULT_CORPUS) -> dict[str, str]:
    """repo name -> blob URL prefix at the pinned commit."""
    manifest = corpus / "manifest.yaml"
    if not manifest.exists():
        return {}
    repos = yaml.safe_load(manifest.read_text(encoding="utf-8"))["repos"]
    return {r["name"]: f"{r['url']}/blob/{r['commit']}" for r in repos}


def make_retriever(name: str, index: VectorIndex, embedder: Embedder) -> Retriever:
    dense = DenseRetriever(index, embedder)
    if name == "dense":
        return dense
    hybrid = HybridRetriever([dense, BM25Retriever(index.chunks)])
    if name == "hybrid":
        return hybrid
    if name == "hybrid+rerank":
        return RerankRetriever(hybrid, FastEmbedCrossEncoder())
    raise ValueError(f"unknown retriever {name!r}")


def create_app(retriever: Retriever, embedder: Embedder, n_chunks: int, min_score: float = 0.0,
               urls: dict[str, str] | None = None) -> FastAPI:
    app = FastAPI(title="ragdocs", description="Questions about my repositories, answered from their docs")
    extractive = RAG(retriever, ExtractiveGenerator(embedder, min_score=min_score))
    urls = urls or {}

    @lru_cache(maxsize=None)
    def llm_rag(backend: str) -> RAG:
        return RAG(retriever, LLMGenerator(make_llm(backend)))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": n_chunks, "embedder": embedder.name, "retriever": retriever.name}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        if req.backend == "extractive":
            rag = extractive
        elif req.backend in BACKENDS:
            try:
                rag = llm_rag(req.backend)
            except RuntimeError as e:  # no key configured for this backend
                raise HTTPException(400, str(e)) from e
        else:
            raise HTTPException(400, f"backend must be extractive or one of {BACKENDS}")
        answer = rag.ask(req.question, req.k)

        def url(doc_id: str) -> str | None:
            repo, _, path = doc_id.partition("/")
            return f"{urls[repo]}/{path}" if repo in urls else None

        return AskResponse(
            answer=answer.text, abstained=answer.abstained, backend=answer.backend, retriever=retriever.name,
            sources=[Source(n=n, chunk_id=h.chunk.chunk_id, heading=h.chunk.heading, score=round(h.score, 4),
                            cited=n in answer.citations, url=url(h.chunk.doc_id))
                     for n, h in enumerate(answer.hits, 1)],
        )

    return app


def app_from_env() -> FastAPI:
    index = VectorIndex.load(Path(os.getenv("RAGDOCS_INDEX", DEFAULT_INDEX)))
    embedder = make_embedder(index.embedder_name)
    retriever = make_retriever(os.getenv("RAGDOCS_RETRIEVER", "hybrid+rerank"), index, embedder)
    return create_app(retriever, embedder, len(index.chunks), default_min_score(), source_urls())


def __getattr__(name: str):  # `uvicorn ragdocs.api:app` builds the app on first access, not on import
    if name == "app":
        return app_from_env()
    raise AttributeError(name)
