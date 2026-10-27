from ragdocs.chunking import chunk_corpus, chunk_fixed
from ragdocs.embeddings import HashingEmbedder
from ragdocs.index import VectorIndex
from ragdocs.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, RerankRetriever


class ListRetriever:
    """A retriever that returns a fixed ranking, to test fusion by hand."""

    def __init__(self, name, chunks):
        self.name = name
        self.chunks = chunks

    def retrieve(self, question, k):
        return [Hit(c, 1.0) for c in self.chunks[:k]]


class FakeCrossEncoder:
    """Scores a chunk by whether it mentions 'PSI'."""

    def score(self, question, texts):
        return [float("PSI" in t) for t in texts]


def test_bm25_prefers_rare_exact_terms(docs):
    bm25 = BM25Retriever(chunk_corpus(docs))
    top = bm25.retrieve("rollback.sh", 1)[0]
    assert top.chunk.heading.endswith("Rollback")
    assert bm25.retrieve("zzzqqq", 3)[0].score == 0.0


def test_rrf_rewards_chunks_both_retrievers_agree_on(docs):
    a, b, c = chunk_corpus(docs)[:3]
    hybrid = HybridRetriever([ListRetriever("x", [a, b, c]), ListRetriever("y", [b, c, a])], rrf_k=60)
    ranked = [h.chunk for h in hybrid.retrieve("q", 3)]
    # b: 1/62 + 1/61 beats a: 1/61 + 1/63 and c: 1/63 + 1/62
    assert ranked == [b, a, c]


def test_rerank_reorders_the_candidates(docs):
    e = HashingEmbedder()
    dense = DenseRetriever(VectorIndex.build(chunk_corpus(docs), e), e)
    reranked = RerankRetriever(dense, FakeCrossEncoder(), candidates=10).retrieve("canary traffic", 1)
    assert "PSI" in reranked[0].chunk.text
    assert reranked[0].score == 1.0


def test_fixed_chunks_ignore_headings(docs):
    chunks = chunk_fixed(docs[0], max_words=10, overlap=2)
    assert all(c.heading == "" for c in chunks)
    assert len(chunks) > 1 and chunks[0].chunk_id == "demo/runbook.md#0"
