"""Command line: build the index and search it.

    python -m ragdocs build                 # chunk + embed the corpus into index/
    python -m ragdocs search "question" -k 5
    python -m ragdocs ask "question" [--backend extractive|ollama|gemini|claude] [--no-grounding]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .chunking import chunk_corpus
from .corpus import DEFAULT_CORPUS, load_corpus
from .embeddings import DEFAULT_MODEL, make_embedder
from .generation import ExtractiveGenerator, LLMGenerator
from .index import VectorIndex
from .llm import BACKENDS, make_llm
from .pipeline import RAG
from .retrieval import DenseRetriever

DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "index"


def cmd_build(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    chunks = chunk_corpus(docs, args.max_words, args.overlap)
    sizes = [len(c.text.split()) for c in chunks]
    print(f"{len(docs)} documents -> {len(chunks)} chunks "
          f"(words per chunk: median {statistics.median(sizes):.0f}, max {max(sizes)}, min {min(sizes)})")
    index = VectorIndex.build(chunks, make_embedder(args.embedder))
    index.save(args.index)
    print(f"index with {index.vectors.shape} vectors from {index.embedder_name} saved to {args.index}")


def cmd_search(args: argparse.Namespace) -> None:
    index = VectorIndex.load(args.index)
    query_vector = make_embedder(index.embedder_name).embed_query(args.question)
    for rank, (chunk, score) in enumerate(index.search(query_vector, args.k), 1):
        preview = " ".join(chunk.text.split())[:160]
        print(f"{rank}. {score:.3f}  {chunk.chunk_id}  [{chunk.heading}]\n   {preview}")


def make_rag(index_dir: Path, backend: str = "extractive", grounded: bool = True, k: int = 5) -> RAG:
    index = VectorIndex.load(index_dir)
    embedder = make_embedder(index.embedder_name)
    if backend == "extractive":
        generator = ExtractiveGenerator(embedder)
    else:
        generator = LLMGenerator(make_llm(backend), grounded)
    return RAG(DenseRetriever(index, embedder), generator, k)


def cmd_ask(args: argparse.Namespace) -> None:
    answer = make_rag(args.index, args.backend, not args.no_grounding, args.k).ask(args.question)
    print(f"{answer.text}\n")
    for n, hit in enumerate(answer.hits, 1):
        mark = "*" if n in answer.citations else " "
        print(f"{mark}[{n}] {hit.score:.3f}  {hit.chunk.chunk_id}  [{hit.chunk.heading}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragdocs")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="chunk and embed the corpus")
    build.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    build.add_argument("--embedder", default=DEFAULT_MODEL)
    build.add_argument("--max-words", type=int, default=220)
    build.add_argument("--overlap", type=int, default=40)
    build.set_defaults(func=cmd_build)

    search = sub.add_parser("search", help="show the top-k chunks for a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--backend", default="extractive", choices=("extractive", *BACKENDS))
    ask.add_argument("--no-grounding", action="store_true", help="drop the grounding instruction (LLM backends)")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)
