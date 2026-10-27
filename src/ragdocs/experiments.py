"""Run every measured experiment and write eval/results/.

    python -m ragdocs eval

1. Retrieval: recall@k and MRR for dense (naive), BM25, hybrid, and each with
   a cross-encoder rerank, on eval/golden.yaml.
2. Answers (extractive, no LLM): the abstention threshold is picked on
   eval/dev.yaml only, then faithfulness, answer hit and abstention are
   measured on eval/golden.yaml.
3. Chunking: structure-aware (the default) against fixed word windows of
   220 words, and huge fixed windows of 1000 words (broken on purpose).
4. k=1 (broken on purpose).

Everything here runs locally with no key. The LLM generator and the
LLM-as-judge are not part of it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .chunking import chunk_corpus, chunk_fixed
from .corpus import load_corpus
from .embeddings import DEFAULT_MODEL, Embedder, make_embedder
from .evaluate import (
    generation_metrics, load_questions, pick_threshold, retrieval_metrics, unmatched_evidence,
)
from .generation import ExtractiveGenerator
from .index import VectorIndex
from .pipeline import RAG
from .retrieval import (
    DEFAULT_RERANKER, BM25Retriever, CrossEncoder, DenseRetriever, FastEmbedCrossEncoder, HybridRetriever,
    RerankRetriever, Retriever,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"


def build_retrievers(index: VectorIndex, embedder: Embedder, cross_encoder: CrossEncoder | None) -> dict:
    dense = DenseRetriever(index, embedder)
    bm25 = BM25Retriever(index.chunks)
    hybrid = HybridRetriever([dense, bm25])
    retrievers: dict[str, Retriever] = {"dense": dense, "bm25": bm25, "hybrid": hybrid}
    if cross_encoder is not None:
        retrievers["dense+rerank"] = RerankRetriever(dense, cross_encoder)
        retrievers["hybrid+rerank"] = RerankRetriever(hybrid, cross_encoder)
    return retrievers


def answer_all(rag: RAG, questions, k: int) -> dict:
    return {q.qid: rag.ask(q.question, k) for q in questions}


def run(corpus: Path | None = None, out_dir: Path = EVAL / "results", embedder_name: str = DEFAULT_MODEL,
        reranker: str | None = DEFAULT_RERANKER, k: int = 5) -> dict:
    t0 = time.time()
    golden = load_questions(EVAL / "golden.yaml")
    dev = load_questions(EVAL / "dev.yaml")
    docs = load_corpus(corpus) if corpus else load_corpus()
    embedder = make_embedder(embedder_name)
    cross_encoder = FastEmbedCrossEncoder(reranker) if reranker else None

    chunks = chunk_corpus(docs)
    missing = unmatched_evidence(golden + dev, chunks)
    if missing:
        raise SystemExit("evidence quotes that no chunk contains:\n  " + "\n  ".join(missing))
    index = VectorIndex.build(chunks, embedder)
    retrievers = build_retrievers(index, embedder, cross_encoder)

    results: dict = {
        "setup": {"embedder": embedder.name, "reranker": reranker, "n_docs": len(docs), "n_chunks": len(chunks),
                  "golden_answerable": sum(q.answerable for q in golden),
                  "golden_unanswerable": sum(not q.answerable for q in golden),
                  "dev": len(dev), "k": k},
        "retrieval": {name: retrieval_metrics(r, golden) for name, r in retrievers.items()},
    }

    # Answers use the advanced pipeline, chosen before any result; the threshold is picked on dev only.
    best = "hybrid+rerank" if "hybrid+rerank" in retrievers else "hybrid"
    probe = RAG(retrievers[best], ExtractiveGenerator(embedder, min_score=float("-inf")), k)
    dev_scores = {qid: a.confidence for qid, a in answer_all(probe, dev, k).items()}
    threshold = pick_threshold(dev_scores, dev)
    rag = RAG(retrievers[best], ExtractiveGenerator(embedder, min_score=threshold), k)
    answers = answer_all(rag, golden, k)
    results["answers"] = {"retriever": best, "threshold_from_dev": threshold, "dev_scores": dev_scores,
                          f"k={k}": generation_metrics(answers, golden),
                          "k=1": generation_metrics(answer_all(rag, golden, 1), golden)}
    results["examples"] = {qid: {"question": a.question, "answer": a.text, "cited": a.cited_chunks}
                           for qid, a in list(answers.items())[:6]}

    # Chunking: fixed windows, normal and huge (broken on purpose). Dense retrieval only, to isolate chunking.
    results["chunking"] = {"structure-aware 220": results["retrieval"]["dense"]}
    for label, words, overlap in (("fixed 220", 220, 40), ("fixed 1000", 1000, 100)):
        fixed = [c for d in docs for c in chunk_fixed(d, words, overlap)]
        fixed_index = VectorIndex.build(fixed, embedder)
        fixed_dense = DenseRetriever(fixed_index, embedder)
        results["chunking"][label] = retrieval_metrics(fixed_dense, golden) | {"n_chunks": len(fixed)}
        if label == "fixed 1000":
            huge_rag = RAG(fixed_dense, ExtractiveGenerator(embedder, min_score=threshold), k)
            results["answers"]["huge chunks"] = generation_metrics(answer_all(huge_rag, golden, k), golden)
    results["seconds"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render(results), encoding="utf-8")
    return results


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _retrieval_table(rows: dict) -> list[str]:
    lines = ["| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | recall@5 paraphrase | recall@5 exact |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        a, kinds = r["all"], r["by_kind"]
        lines.append(f"| {name} | " + " | ".join(_pct(a[f"recall@{k}"]) for k in (1, 3, 5, 10))
                     + f" | {a['mrr@10']:.2f} | {_pct(kinds.get('paraphrase', {}).get('recall@5'))}"
                     + f" | {_pct(kinds.get('exact', {}).get('recall@5'))} |")
    return lines


def _answer_row(label: str, g: dict) -> str:
    return (f"| {label} | {_pct(g['faithfulness'])} (n={g['faithfulness_n']}) | {_pct(g['answer_hit'])} | "
            f"{_pct(g['false_abstention'])} | {_pct(g['correct_abstention'])} |")


def render(r: dict) -> str:
    s, a = r["setup"], r["answers"]
    lines = [
        "# Evaluation report", "",
        f"Generated by `python -m ragdocs eval` in {r['seconds']} s. Embedder `{s['embedder']}`, "
        f"reranker `{s['reranker']}`, {s['n_docs']} documents, {s['n_chunks']} chunks. "
        f"Golden set: {s['golden_answerable']} answerable + {s['golden_unanswerable']} unanswerable questions; "
        f"dev set: {s['dev']} questions.", "",
        "## Retrieval", "", *_retrieval_table(r["retrieval"]), "",
        f"## Answers (extractive, no LLM, retriever `{a['retriever']}`)", "",
        f"Abstention threshold {a['threshold_from_dev']} picked on the dev set only.", "",
        "| Setting | Faithfulness (lexical) | Answer hit | Abstained, answerable | Abstained, unanswerable |",
        "|---|---|---|---|---|",
        _answer_row(f"k={s['k']}", a[f"k={s['k']}"]),
        _answer_row("k=1 (broken on purpose)", a["k=1"]),
        _answer_row(f"fixed 1000-word chunks, dense, k={s['k']} (broken on purpose)", a["huge chunks"]), "",
        "Extractive answers copy sentences from the retrieved chunks, so lexical faithfulness is ~100% by "
        "construction. It becomes informative with an LLM generator.", "",
        "## Chunking (dense retrieval only)", "",
        *_retrieval_table(r["chunking"]), "",
    ]
    return "\n".join(lines)
