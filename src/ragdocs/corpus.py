"""Load the corpus: every markdown file under corpus/<repo>/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "corpus"


@dataclass(frozen=True)
class Document:
    doc_id: str  # "<repo>/<path>", e.g. "invoice-mlops/docs/runbook.md"
    repo: str
    path: str
    text: str


def load_corpus(root: Path = DEFAULT_CORPUS) -> list[Document]:
    docs = []
    for file in sorted(root.rglob("*.md")):
        rel = file.relative_to(root).as_posix()
        repo, _, path = rel.partition("/")
        if not path:  # a file directly under corpus/ (e.g. a README about the corpus)
            continue
        docs.append(Document(rel, repo, path, file.read_text(encoding="utf-8")))
    return docs
