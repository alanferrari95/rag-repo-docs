"""Split markdown documents into chunks for retrieval.

Strategy, in order:
1. Cut at headings. A section is the natural unit of topic in these docs.
2. Inside a section, split into blocks: paragraphs, lists, tables and fenced
   code blocks each stay whole.
3. Pack blocks into chunks of at most `max_words`. A block longer than that is
   cut into overlapping word windows. When a section needs several chunks, the
   next chunk starts with the last block of the previous one if that block is
   short, so a sentence that bridges two chunks is in both.

Every chunk keeps its heading path ("Rollout > Canary"), and the text that is
embedded is prefixed with "<repo> > <path> > <heading path>", so a paragraph
like "run this command" still says what it is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<n>"
    doc_id: str
    heading: str  # heading path inside the document, "" before the first heading
    text: str

    @property
    def embed_text(self) -> str:
        repo, _, path = self.doc_id.partition("/")
        crumbs = " > ".join(p for p in (repo, path, self.heading) if p)
        return f"{crumbs}\n\n{self.text}"


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Return (heading path, body lines) per section, ignoring '#' inside code fences."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in body):
            sections.append((" > ".join(title for _, title in stack), body.copy()))
        body.clear()

    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            stack = [(lvl, t) for lvl, t in stack if lvl < level]
            stack.append((level, match.group(2)))
        else:
            body.append(line)
    flush()
    return sections


def _blocks(lines: list[str]) -> list[str]:
    """Split section lines into blocks separated by blank lines; code fences stay whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _windows(block: str, max_words: int, overlap: int) -> list[str]:
    words = block.split()
    step = max(1, max_words - overlap)
    return [" ".join(words[i:i + max_words]) for i in range(0, max(1, len(words) - overlap), step)]


def _n_words(text: str) -> int:
    return len(text.split())


def chunk_document(doc: Document, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, lines in _sections(doc.text):
        pieces: list[str] = []
        for block in _blocks(lines):
            pieces.extend(_windows(block, max_words, overlap) if _n_words(block) > max_words else [block])

        packed: list[list[str]] = [[]]
        for piece in pieces:
            current = packed[-1]
            if current and sum(map(_n_words, current)) + _n_words(piece) > max_words:
                carry = current[-1] if _n_words(current[-1]) <= overlap else None
                packed.append([carry] if carry else [])
            packed[-1].append(piece)

        for group in packed:
            if group:
                n = len(chunks)
                chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, heading, "\n\n".join(group)))
    return chunks


def chunk_corpus(docs: list[Document], max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, max_words, overlap)]
