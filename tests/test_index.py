import numpy as np

from ragdocs.chunking import chunk_corpus
from ragdocs.embeddings import HashingEmbedder
from ragdocs.index import VectorIndex


def test_hashing_embedder_is_unit_length_and_deterministic():
    e = HashingEmbedder(64)
    a, b = e.embed_documents(["canary rollback", "canary rollback"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.isclose(np.linalg.norm(e.embed_query("")), 0.0)  # empty text: zero vector, no NaN


def test_search_ranks_the_matching_chunk_first(docs):
    e = HashingEmbedder()
    index = VectorIndex.build(chunk_corpus(docs), e)
    hits = index.search(e.embed_query("PSI drift alert"), k=2)
    assert hits[0][0].doc_id == "demo/drift.md"
    assert hits[0][1] >= hits[1][1]


def test_k_larger_than_index_returns_everything(docs):
    e = HashingEmbedder()
    index = VectorIndex.build(chunk_corpus(docs), e)
    assert len(index.search(e.embed_query("traffic"), k=100)) == len(index.chunks)


def test_save_and_load_round_trip(docs, tmp_path):
    e = HashingEmbedder()
    index = VectorIndex.build(chunk_corpus(docs), e)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.chunks == index.chunks
    assert loaded.embedder_name == "hashing-512"
    assert np.array_equal(loaded.vectors, index.vectors)
