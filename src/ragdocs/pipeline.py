"""The query phase in one place: retrieve the top-k, then generate a cited answer."""

from __future__ import annotations

from typing import Protocol

from .generation import Answer
from .retrieval import Hit, Retriever


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Answer: ...


class RAG:
    def __init__(self, retriever: Retriever, generator: Generator, k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retriever.retrieve(question, k or self.k)
        return self.generator.generate(question, hits)
