import pytest

from ragdocs.chunking import chunk_corpus
from ragdocs.embeddings import HashingEmbedder
from ragdocs.evaluate import (
    Question, answer_hit, faithfulness, generation_metrics, is_relevant, load_questions, pick_threshold,
    retrieval_metrics, unmatched_evidence,
)
from ragdocs.generation import ABSTAIN, Answer
from ragdocs.index import VectorIndex
from ragdocs.judge import LLMJudge, parse_judgement
from ragdocs.retrieval import DenseRetriever, Hit

DRIFT_Q = Question("q1", "Which statistics does the drift job compute?", "paraphrase", "PSI and KS",
                   (("demo/drift.md", "computes PSI and KS on every feature"),))
CANARY_Q = Question("q2", "canary.sh", "exact", "20%", (("demo/runbook.md", "sends 20% of traffic"),))
NOPE_Q = Question("u1", "What is the refund policy?", "unanswerable", None, ())


@pytest.fixture
def chunks(docs):
    return chunk_corpus(docs)


def _hits(chunks):
    return [Hit(c, 1.0) for c in chunks]


def test_relevance_is_doc_plus_quote_whitespace_insensitive(chunks):
    drift = [c for c in chunks if c.doc_id == "demo/drift.md"][0]
    assert is_relevant(drift, DRIFT_Q)
    spaced = Question("q", "", "paraphrase", None, (("demo/drift.md", "computes  PSI\nand KS"),))
    assert is_relevant(drift, spaced)
    other_doc = Question("q", "", "paraphrase", None, (("demo/runbook.md", "computes PSI and KS"),))
    assert not is_relevant(drift, other_doc)


def test_unmatched_evidence_reports_typos(chunks):
    typo = Question("q9", "", "exact", None, (("demo/drift.md", "computes PSI and KL"),))
    assert unmatched_evidence([DRIFT_Q, typo], chunks) == ["q9: demo/drift.md: 'computes PSI and KL'"]


def test_retrieval_metrics_recall_and_mrr(chunks):
    e = HashingEmbedder()
    dense = DenseRetriever(VectorIndex.build(chunks, e), e)
    m = retrieval_metrics(dense, [DRIFT_Q, CANARY_Q, NOPE_Q], ks=(1, 3))
    assert m["all"]["n"] == 2  # the unanswerable question does not count
    assert m["all"]["recall@3"] == 1.0
    ranks = m["ranks"]
    assert m["all"]["mrr@10"] == pytest.approx((1 / ranks["q1"] + 1 / ranks["q2"]) / 2)
    assert set(m["by_kind"]) == {"paraphrase", "exact"}


def test_faithfulness_flags_an_unsupported_sentence(chunks):
    hits = _hits([c for c in chunks if c.doc_id == "demo/drift.md"])
    good = Answer("q", "The drift job computes PSI and KS on every feature [1].", hits)
    mixed = Answer("q", "The drift job computes PSI and KS [1]. It retrains the model on BigQuery ML every hour.",
                   hits)
    assert faithfulness(good) == 1.0
    assert faithfulness(mixed) == 0.5
    assert faithfulness(Answer("q", ABSTAIN, hits)) is None


def test_answer_hit_and_abstention_metrics(chunks):
    hits = _hits(chunks)
    answers = {
        "q1": Answer("q", "The drift job computes PSI and KS on every feature [1].", hits),
        "q2": Answer("q", ABSTAIN, hits),
        "u1": Answer("q", ABSTAIN, hits),
    }
    assert answer_hit(answers["q1"], DRIFT_Q)
    g = generation_metrics(answers, [DRIFT_Q, CANARY_Q, NOPE_Q])
    assert g["answer_hit"] == 0.5
    assert g["false_abstention"] == 0.5
    assert g["correct_abstention"] == 1.0
    assert g["faithfulness"] == 1.0 and g["faithfulness_n"] == 1


def test_pick_threshold_separates_answerable_from_not():
    qs = [Question("a", "", "paraphrase", None, (("d", "x"),)), Question("b", "", "exact", None, (("d", "x"),)),
          Question("u", "", "unanswerable", None, ())]
    assert pick_threshold({"a": 0.8, "b": 0.6, "u": 0.3}, qs) == 0.45


def test_load_questions_reads_the_yaml_format(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("questions:\n  - id: q1\n    question: Why?\n    kind: exact\n    answer: Because\n"
                    "    evidence:\n      - doc: a/b.md\n        quote: because so\n"
                    "  - id: u1\n    question: Nope?\n    kind: unanswerable\n    answer: null\n    evidence: []\n",
                    encoding="utf-8")
    q1, u1 = load_questions(path)
    assert q1.evidence == (("a/b.md", "because so"),) and q1.answerable
    assert not u1.answerable


class FakeJudgeLLM:
    name = "fake"

    def complete(self, system, user):
        assert "Answer to check:" in user and "<context>" in user
        return ('Sure: {"claims": [{"claim": "computes PSI", "verdict": "supported"}, '
                '{"claim": "retrains hourly", "verdict": "unsupported"}]}')


def test_llm_judge_scores_claims(chunks):
    answer = Answer("q", "It computes PSI [1]. It retrains hourly.", _hits(chunks[:1]))
    judgement = LLMJudge(FakeJudgeLLM()).judge(answer)
    assert judgement.faithfulness == 0.5
    assert LLMJudge(FakeJudgeLLM()).judge(Answer("q", ABSTAIN, [])) is None


def test_judge_rejects_malformed_replies():
    with pytest.raises(ValueError, match="no JSON"):
        parse_judgement("I think it is fine.")
    with pytest.raises(ValueError, match="unknown verdict"):
        parse_judgement('{"claims": [{"claim": "x", "verdict": "maybe"}]}')


def test_answer_sentences_split_after_citations_without_periods():
    from ragdocs.evaluate import answer_sentences
    text = "Roll back with rollback.sh; no rebuild [1] Checked on 2026-10-22: [2] The model is Flash. [3]"
    assert answer_sentences(text) == ["Roll back with rollback.sh; no rebuild [1]", "Checked on 2026-10-22: [2]",
                                      "The model is Flash."]


def test_extractive_answer_spanning_two_chunks_is_fully_faithful(chunks):
    hits = _hits(chunks)
    text = "`canary.sh` sends 20% of traffic to the new revision [2] The drift job computes PSI and KS on every feature [4]"
    assert faithfulness(Answer("q", text, hits)) == 1.0
