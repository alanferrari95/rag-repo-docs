"""Evaluate retrieval and generation separately, on a hand-made golden set.

Retrieval (needs no LLM):
- recall@k (hit rate): share of answerable questions with at least one
  relevant chunk in the top k. A chunk is relevant if it comes from the
  evidence document and contains the evidence quote, so the labels survive a
  change of chunking.
- MRR@10: 1 / rank of the first relevant chunk (0 if not in the top 10).

Generation:
- faithfulness (offline, lexical): share of answer sentences whose content
  words are at least SUPPORT_THRESHOLD covered by one retrieved chunk. An
  extractive answer copies sentences, so it scores ~1 by construction; the
  check matters for LLM answers. judge.py has the LLM-as-judge version.
- answer hit: the answer contains (most content words of) an evidence quote.
- abstention: "I don't know" on unanswerable questions (good) and on
  answerable ones (bad).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .chunking import Chunk
from .generation import ABSTAIN, CITATION, SENTENCE_END, Answer
from .retrieval import Retriever, tokenize

KS = (1, 3, 5, 10)
SUPPORT_THRESHOLD = 0.8
STOPWORDS = set("""a an and are as at be by can do does for from has have how if in into is it its of on or so
that the their then there these this to was were what when where which who why will with without you your""".split())


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    kind: str  # paraphrase | exact | unanswerable
    answer: str | None
    evidence: tuple[tuple[str, str], ...]  # (doc_id, quote)

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Question(q["id"], q["question"], q["kind"], q.get("answer"),
                     tuple((e["doc"], e["quote"]) for e in q.get("evidence") or []))
            for q in data["questions"]]


def _norm(text: str) -> str:
    return " ".join(text.split())


def is_relevant(chunk: Chunk, question: Question) -> bool:
    return any(chunk.doc_id == doc and _norm(quote) in _norm(chunk.text) for doc, quote in question.evidence)


def unmatched_evidence(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Evidence that no chunk contains: a typo in the golden set, or a quote cut in two by the chunker."""
    missing = []
    for q in questions:
        for doc, quote in q.evidence:
            if not any(c.doc_id == doc and _norm(quote) in _norm(c.text) for c in chunks):
                missing.append(f"{q.qid}: {doc}: {quote!r}")
    return missing


# ------------------------------------------------------------------ retrieval

def first_relevant_rank(retriever: Retriever, question: Question, depth: int) -> int | None:
    for rank, hit in enumerate(retriever.retrieve(question.question, depth), 1):
        if is_relevant(hit.chunk, question):
            return rank
    return None


def retrieval_metrics(retriever: Retriever, questions: list[Question], ks=KS) -> dict:
    answerable = [q for q in questions if q.answerable]
    depth = max(max(ks), 10)
    ranks = {q.qid: first_relevant_rank(retriever, q, depth) for q in answerable}

    def summarize(qs: list[Question]) -> dict:
        r = [ranks[q.qid] for q in qs]
        out = {f"recall@{k}": sum(x is not None and x <= k for x in r) / len(r) for k in ks}
        out["mrr@10"] = sum(1 / x for x in r if x is not None and x <= 10) / len(r)
        out["n"] = len(r)
        return out

    by_kind = {kind: summarize([q for q in answerable if q.kind == kind])
               for kind in sorted({q.kind for q in answerable})}
    return {"retriever": retriever.name, "all": summarize(answerable), "by_kind": by_kind, "ranks": ranks}


# ------------------------------------------------------------------ generation

def content_words(text: str) -> set[str]:
    return {t for t in tokenize(CITATION.sub(" ", text)) if t not in STOPWORDS}


def sentence_support(sentence: str, chunks: list[Chunk]) -> float:
    """Best share of the sentence's content words found in a single chunk."""
    words = content_words(sentence)
    if not words:
        return 1.0
    return max((len(words & content_words(c.embed_text)) / len(words) for c in chunks), default=0.0)


def answer_sentences(text: str) -> list[str]:
    parts = SENTENCE_END.split(" ".join(text.split()))
    return [p for p in parts if content_words(p)]


def faithfulness(answer: Answer, threshold: float = SUPPORT_THRESHOLD) -> float | None:
    """Share of answer sentences supported by the retrieved chunks. None for an abstention."""
    if answer.abstained:
        return None
    sentences = answer_sentences(answer.text)
    if not sentences:
        return None
    chunks = [h.chunk for h in answer.hits]
    return sum(sentence_support(s, chunks) >= threshold for s in sentences) / len(sentences)


def answer_hit(answer: Answer, question: Question, threshold: float = SUPPORT_THRESHOLD) -> bool:
    """The answer carries the evidence: most content words of an evidence quote appear in it."""
    said = content_words(answer.text)
    return any(len(content_words(quote) & said) >= threshold * len(content_words(quote))
               for _, quote in question.evidence)


def generation_metrics(answers: dict[str, Answer], questions: list[Question]) -> dict:
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    faith = [f for q in questions if (f := faithfulness(answers[q.qid])) is not None]
    return {
        "faithfulness": float(np.mean(faith)) if faith else None,
        "faithfulness_n": len(faith),
        "answer_hit": sum(answer_hit(answers[q.qid], q) for q in answerable) / len(answerable),
        "false_abstention": sum(answers[q.qid].abstained for q in answerable) / len(answerable),
        "correct_abstention": (sum(answers[q.qid].abstained for q in unanswerable) / len(unanswerable)
                               if unanswerable else None),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }


def pick_threshold(best_scores: dict[str, float], questions: list[Question]) -> float:
    """Abstention threshold that best separates answerable from unanswerable questions (dev set only)."""
    labels = {q.qid: q.answerable for q in questions}
    candidates = sorted(best_scores.values())
    cuts = [(a + b) / 2 for a, b in zip(candidates, candidates[1:])] or candidates

    def accuracy(t: float) -> float:
        return sum((s >= t) == labels[qid] for qid, s in best_scores.items()) / len(best_scores)

    return round(max(cuts, key=lambda t: (accuracy(t), -t)), 3)


SLUG = re.compile(r"[^a-z0-9+]+")


def slug(name: str) -> str:
    return SLUG.sub("-", name.lower()).strip("-")
