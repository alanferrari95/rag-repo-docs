import pytest
from fastapi.testclient import TestClient

from ragdocs.api import create_app, source_urls
from ragdocs.chunking import chunk_corpus
from ragdocs.embeddings import HashingEmbedder
from ragdocs.index import VectorIndex
from ragdocs.retrieval import DenseRetriever


@pytest.fixture
def client(docs):
    e = HashingEmbedder()
    index = VectorIndex.build(chunk_corpus(docs), e)
    app = create_app(DenseRetriever(index, e), e, len(index.chunks), min_score=0.2,
                     urls={"demo": "https://github.com/me/demo/blob/abc123"})
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 4, "embedder": "hashing-2048",
                                            "retriever": "dense"}


def test_ask_returns_a_cited_answer_with_links(client):
    body = client.post("/ask", json={"question": "Which statistics does the drift job compute?", "k": 2}).json()
    assert "PSI and KS" in body["answer"] and not body["abstained"]
    cited = [s for s in body["sources"] if s["cited"]]
    assert cited and cited[0]["url"] == "https://github.com/me/demo/blob/abc123/drift.md"
    assert [s["n"] for s in body["sources"]] == [1, 2]


def test_ask_abstains_when_nothing_is_close(client):
    body = client.post("/ask", json={"question": "Quelle est la capitale de la France ?"}).json()
    assert body["abstained"] and body["answer"].startswith("I don't know")


def test_llm_backend_without_key_is_a_400(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "What does the drift job compute?", "backend": "gemini"})
    assert r.status_code == 400 and "GEMINI_API_KEY" in r.json()["detail"]


def test_bad_requests_are_rejected(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "backend": "gpt"}).status_code == 400


def test_source_urls_come_from_the_manifest():
    urls = source_urls()
    assert urls["invoice-mlops"].startswith("https://github.com/alanferrari95/invoice-mlops/blob/")
    assert set(urls) == {"invoice-mlops", "agent-harness", "jobbot"}
