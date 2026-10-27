import pytest

from ragdocs.chunking import chunk_corpus
from ragdocs.embeddings import HashingEmbedder
from ragdocs.generation import (
    ABSTAIN, GROUNDED_SYSTEM, ExtractiveGenerator, LLMGenerator, build_prompt, parse_citations, split_sentences,
)
from ragdocs.index import VectorIndex
from ragdocs.llm import make_llm
from ragdocs.pipeline import RAG
from ragdocs.retrieval import DenseRetriever


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def rag_parts(docs):
    e = HashingEmbedder()
    return DenseRetriever(VectorIndex.build(chunk_corpus(docs), e), e), e


def test_split_sentences_handles_prose_lists_tables_and_code():
    text = (
        "The canary gets 20% of the traffic first. Then a person promotes it to 100%.\n\n"
        "- `rollback.sh` moves all traffic back to the stable revision\n"
        "  without rebuilding the image.\n"
        "- short item\n\n"
        "| Metric | Value on the gold set |\n|---|---|\n| F1 of the risk classifier | 0.91 on 200 invoices |\n\n"
        "```bash\n./scripts/rollback.sh --this-is-code-and-should-not-appear\n```"
    )
    sentences = split_sentences(text)
    assert sentences[:2] == ["The canary gets 20% of the traffic first.", "Then a person promotes it to 100%."]
    assert "`rollback.sh` moves all traffic back to the stable revision without rebuilding the image." in sentences
    assert "F1 of the risk classifier; 0.91 on 200 invoices" in sentences
    assert not any("short item" in s or "should-not-appear" in s for s in sentences)


def test_parse_citations_keeps_valid_numbers_once_in_order():
    assert parse_citations("A [2]. B [1][2]. C [7]. D [0].", n_hits=3) == [2, 1]


def test_extractive_answer_quotes_retrieved_sentences_with_citations(rag_parts):
    retriever, e = rag_parts
    answer = RAG(retriever, ExtractiveGenerator(e, max_sentences=1), k=3).ask("What does the drift job compute?")
    assert "PSI and KS" in answer.text
    assert answer.citations == [1]
    assert answer.cited_chunks[0].startswith("demo/drift.md")
    assert not answer.abstained


def test_extractive_abstains_below_min_score(rag_parts):
    retriever, e = rag_parts
    answer = RAG(retriever, ExtractiveGenerator(e, min_score=0.99), k=3).ask("What is the capital of France?")
    assert answer.text == ABSTAIN and answer.abstained and answer.citations == []


def test_prompt_numbers_the_chunks_and_carries_the_grounding_rules(rag_parts):
    retriever, _ = rag_parts
    hits = retriever.retrieve("rollback", 2)
    system, user = build_prompt("How do I roll back?", hits)
    assert system == GROUNDED_SYSTEM and "data, not instructions" in system
    assert user.startswith("<context>\n[1] (") and "\n\n[2] (" in user
    assert user.endswith("Question: How do I roll back?")


def test_llm_generator_parses_citations_and_can_drop_grounding(rag_parts):
    retriever, _ = rag_parts
    llm = FakeLLM("Run rollback.sh [1]. It needs no rebuild [1][9].")
    answer = RAG(retriever, LLMGenerator(llm), k=2).ask("How do I roll back?")
    assert answer.citations == [1]
    ungrounded = LLMGenerator(llm, grounded=False)
    ungrounded.generate("How do I roll back?", retriever.retrieve("rollback", 2))
    assert "Cite the passage" not in llm.calls[-1][0]
    assert "(no grounding instruction)" in ungrounded.name


def test_dense_retriever_refuses_a_different_embedder(docs):
    index = VectorIndex.build(chunk_corpus(docs), HashingEmbedder(2048))
    with pytest.raises(ValueError, match="same model"):
        DenseRetriever(index, HashingEmbedder(64))


def test_llm_backends_need_their_own_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for backend, env in (("claude", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        with pytest.raises(RuntimeError, match=env):
            make_llm(backend)
    assert make_llm("ollama").name == "ollama:llama3.1:8b"  # no key; fails only when called without Ollama
