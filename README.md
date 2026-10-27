# rag-repo-docs

A small retrieval-augmented generation (RAG) system that answers questions about three of my
own repositories ([invoice-mlops](https://github.com/alanferrari95/invoice-mlops),
[agent-harness](https://github.com/alanferrari95/agent-harness),
[jobbot](https://github.com/alanferrari95/jobbot)) from their documentation, and cites the
passage behind every sentence.

I built it to learn RAG properly, so the point is the measurement, not the chatbot:
retrieval and generation are evaluated separately, on a question set written from the
docs before any retrieval result existed, and every number below comes from
`python -m ragdocs eval`. Everything runs locally on a CPU with no API key. LLM backends
are optional and bring their own key or machine.

## Status: what ran and what is only built

| Piece | Code + tests | Ran |
|---|---|---|
| Corpus snapshot of 16 docs at pinned commits | ✅ | ✅ |
| Structure-aware chunking, `bge-small-en-v1.5` embeddings, numpy index | ✅ | ✅ |
| Dense, BM25, hybrid (RRF) and cross-encoder rerank retrieval | ✅ | ✅ measured below |
| Extractive cited answer with abstention (default, no LLM) | ✅ | ✅ measured below |
| FastAPI `/ask` and `/health` | ✅ | ✅ locally with uvicorn |
| Lexical faithfulness check | ✅ | ✅ on extractive answers, where it is ~100% by construction |
| LLM generation: Ollama, Gemini, Claude | ✅ (fake LLM in tests) | — built, not run |
| Faithfulness with an LLM-as-judge | ✅ (fake LLM in tests) | — built, not run |
| "No grounding instruction" experiment | ✅ (`--no-grounding`) | — built, not run (needs an LLM) |

## Results

Measured on 2026-10-27 with `python -m ragdocs eval` (about 11 minutes on a laptop CPU; full
report in [`eval/results/report.md`](eval/results/report.md)). There are 36 answerable
questions, so one question is worth about 3 points.

**Retrieval: naive vs. advanced**

| Retriever | recall@1 | recall@5 | recall@10 | MRR@10 | recall@5, paraphrase (24) | recall@5, exact (12) |
|---|---|---|---|---|---|---|
| dense (naive RAG) | 58% | 92% | 94% | 0.72 | 92% | 92% |
| BM25 | 64% | 81% | 86% | 0.71 | 71% | 100% |
| hybrid (RRF) | 67% | 94% | 100% | 0.76 | 92% | 100% |
| hybrid + cross-encoder rerank | **75%** | 92% | 97% | **0.82** | 88% | 100% |

- The reranker is what moves the first result: recall@1 goes from 58% to 75% (6 more
  questions with the right chunk first) and MRR@10 from 0.72 to 0.82. Recall@5 does not
  change: with ~100 chunks, the right one was usually in the top 5 already.
- BM25 and dense fail on different questions: BM25 finds every exact identifier but misses
  paraphrases (71%); dense does the opposite. Hybrid search gets both, and every question has a
  relevant chunk in its top 10.

**Chunking (dense retrieval only)**

| Chunking | chunks | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|---|
| by markdown heading, ≤220 words (default) | 127 | 58% | 92% | 0.72 |
| fixed 220-word windows | 93 | 50% | 89% | 0.66 |
| fixed 1000-word windows (broken on purpose) | 25 | 61% | 89% | 0.72 |

Recall is not where huge chunks hurt: with only 25 chunks, almost any chunk of the right
file contains the quote, so recall@1 is even slightly higher (61% vs 58%). The cost shows
up in the answer: answer hit drops from 58% to
47% because the right sentence is buried in 1,000 words.

**Answers (extractive, no LLM, hybrid + rerank, k=5)**

| Setting | Answer hit | Abstained, answerable | Abstained, unanswerable | Faithfulness (lexical) |
|---|---|---|---|---|
| k=5 | 58% | 0% | 0% of 6 | 100% |
| k=1 (broken on purpose) | 56% | 19% | 33% of 6 | 100% |
| 1000-word chunks (broken on purpose) | 47% | 6% | 0% of 6 | 100% |

- **Answer hit is the weak spot.** Retrieval finds the right chunk, but picking sentences by
  similarity often quotes the section's first line instead of the sentence with the
  reason. Synthesizing an answer is the LLM's job, and that path has not been run.
- **Abstention by threshold only works for off-topic questions.** The threshold (0.683),
  picked on the dev set, abstains on 2/2 off-topic questions there. It catches 1/4 of the
  dev set's near-miss questions about these projects (p99 latency, an on-call channel) and
  0/6 in the golden set: they score as high as real ones, both by similarity and by
  reranker score. Abstaining on those needs a generator that reads the passages.
- **Faithfulness is 100% by construction** for extractive answers. The check is still
  useful: it was at 95% on the first run, which exposed a bug in how the check split
  answers into sentences (fixed, with a test).

## How it works

```
INDEXING (once, and when the docs change)
  corpus/*.md ──► chunk by heading ──► embed "repo > file > heading + text" ──► numpy matrix + chunks.jsonl

QUERY (every question)
  question ──► dense top-50 ─┐
           └─► BM25 top-50 ──┴─► RRF fusion ──► cross-encoder rerank ──► top-k ──► answer with [n] citations
                                                                                  (or "I don't know")
```

| Step | File | What it does |
|---|---|---|
| Corpus | [`corpus/manifest.yaml`](corpus/manifest.yaml), [`scripts/snapshot_corpus.py`](scripts/snapshot_corpus.py) | Copies each doc with `git show <commit>:<path>`, so the corpus and the numbers are reproducible |
| Chunking | [`chunking.py`](src/ragdocs/chunking.py) | Cuts at markdown headings; long sections become ≤220-word chunks with overlap; code fences and tables stay whole. Each chunk is embedded with its path, e.g. `invoice-mlops > docs/runbook.md > Rollback` |
| Embeddings | [`embeddings.py`](src/ragdocs/embeddings.py) | `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU, 384 dims). Tests use a hashing embedder that needs no download |
| Index | [`index.py`](src/ragdocs/index.py) | Exact search: one matrix-vector product over unit vectors. At ~100 chunks an ANN index (HNSW, ScaNN) would add complexity and gain nothing |
| Retrieval | [`retrieval.py`](src/ragdocs/retrieval.py) | Dense, BM25, hybrid with reciprocal rank fusion, and a cross-encoder (`ms-marco-MiniLM-L-6-v2`) that reranks 30 candidates |
| Answer | [`generation.py`](src/ragdocs/generation.py) | Extractive by default: the retrieved sentences closest to the question, each with its `[n]`. Abstains below a similarity threshold picked on a separate dev set. The LLM generator uses a grounding prompt that treats the passages as data, not instructions |
| Evaluation | [`evaluate.py`](src/ragdocs/evaluate.py), [`judge.py`](src/ragdocs/judge.py), [`experiments.py`](src/ragdocs/experiments.py) | recall@k and MRR for retrieval; faithfulness, answer hit and abstention for answers; LLM-as-judge |
| API | [`api.py`](src/ragdocs/api.py) | `POST /ask`, with links to each source file at its pinned commit |

## Evaluation method

- **Golden set** ([`eval/golden.yaml`](eval/golden.yaml)): 42 questions drafted with an AI
  assistant that read only the docs, never the code or any retrieval output, so the questions
  could not be shaped by what retrieval returns. Each one has the answer and one or more evidence
  quotes. A retrieved chunk counts as relevant if it comes from the evidence file and
  contains the quote, so the labels survive a change of chunking. Questions are tagged
  `paraphrase` (different words than the docs), `exact` (hinges on an identifier such as a
  script name or env var) or `unanswerable`.
- **Dev set** ([`eval/dev.yaml`](eval/dev.yaml)): a separate, small set used only to pick the
  abstention threshold, so that threshold is never tuned on the questions it is scored on.
- **Retrieval:** recall@k (did a relevant chunk make the top k?) and MRR@10 (how high was the
  first one?).
- **Generation:** faithfulness (share of answer sentences whose content words are covered by
  one retrieved chunk, with an LLM-as-judge version in `judge.py`), answer hit (the answer
  carries the evidence) and abstention on answerable and unanswerable questions.
- **Small n:** with a few dozen questions, one question moves recall by about 3 points. Read
  differences of a few points as noise.

## Run it

With Python 3.11+ (developed on 3.14, Windows, CPU only):

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
```

```bash
python -m ragdocs build
```

```bash
python -m ragdocs ask "How does rollback choose which revision gets the traffic back?"
```

```bash
python -m ragdocs eval
```

```bash
uvicorn ragdocs.api:app
```

`build` downloads the embedding model (~70 MB) once; `eval` also downloads the reranker
(~80 MB) and writes [`eval/results/report.md`](eval/results/report.md). The tests need neither:

```bash
pytest
```

Optional LLM backends, each with your own key or machine (`RAGDOCS_LLM_MODEL` overrides the
model):

| Backend | Needs |
|---|---|
| `--backend ollama` | Ollama running locally (`llama3.1:8b` by default) |
| `--backend gemini` | `GEMINI_API_KEY` |
| `--backend claude` | `ANTHROPIC_API_KEY` |

## Limitations

- The corpus is small (16 docs, ~16k words), so exact search is enough and the numbers come
  from a few dozen questions. It shows the method, not production scale.
- The extractive answer can only quote; it cannot combine two facts into one sentence. Its
  faithfulness is ~100% by construction, so it says nothing about hallucination. That needs
  an LLM generator, which is built but has not been run.
- The LLM-as-judge has not been validated against human labels yet; until it is, its number
  would not be trustworthy either.
- No metadata filters or permissions: every doc here is public.

## License

MIT. The documents under `corpus/` are copied from my own repositories.
