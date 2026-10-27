"""Faithfulness with an LLM-as-judge.

The judge splits the answer into atomic claims and labels each one against
the retrieved context: supported, unsupported (not in the context) or
contradicted. Faithfulness = supported claims / all claims.

It catches what the lexical check in evaluate.py misses: a paraphrase that
is supported but shares few words, or a sentence that reuses the context's
words to say something the context does not say. Before trusting its number,
the judge should be checked against human labels on a sample of answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .generation import Answer, format_context
from .llm import LLM

JUDGE_SYSTEM = """You check whether an answer is supported by the context it was given.
1. Split the answer into short atomic claims (one fact each). Ignore citation markers like [1].
2. For each claim, compare it only with the context, never with your own knowledge:
   - "supported": the context states it, possibly in other words;
   - "unsupported": the context does not say it;
   - "contradicted": the context says something incompatible.
3. Reply with JSON only, no prose:
{"claims": [{"claim": "...", "verdict": "supported" | "unsupported" | "contradicted"}]}"""

VERDICTS = ("supported", "unsupported", "contradicted")
JSON_OBJECT = re.compile(r"\{.*\}", re.S)


@dataclass
class Judgement:
    claims: list[dict]

    @property
    def faithfulness(self) -> float | None:
        if not self.claims:
            return None
        return sum(c["verdict"] == "supported" for c in self.claims) / len(self.claims)


def parse_judgement(reply: str) -> Judgement:
    match = JSON_OBJECT.search(reply)
    if not match:
        raise ValueError(f"judge reply has no JSON object: {reply[:200]!r}")
    claims = json.loads(match.group(0)).get("claims", [])
    for c in claims:
        if c.get("verdict") not in VERDICTS:
            raise ValueError(f"unknown verdict in judge reply: {c!r}")
    return Judgement(claims)


class LLMJudge:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"judge:{llm.name}"

    def judge(self, answer: Answer) -> Judgement | None:
        """None for an abstention: there is nothing to check."""
        if answer.abstained:
            return None
        user = f"{format_context(answer.hits)}\n\nQuestion: {answer.question}\n\nAnswer to check:\n{answer.text}"
        return parse_judgement(self.llm.complete(JUDGE_SYSTEM, user))
