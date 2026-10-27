"""Optional LLM backends. Each brings its own key or its own machine; none is needed to run the project.

- ollama: a local model through Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1), no key.
- gemini: Google's Gemini through its OpenAI-compatible endpoint, with your key in GEMINI_API_KEY.
- claude: Anthropic's Messages API, with your key in ANTHROPIC_API_KEY.
RAGDOCS_LLM_MODEL overrides the model of any backend.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

PRESETS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1:8b", "key_env": None},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.8-flash",
               "key_env": "GEMINI_API_KEY"},
}
CLAUDE_MODEL = "claude-haiku-5-5"
BACKENDS = ("ollama", "gemini", "claude")
TIMEOUT = 120.0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, name: str = "openai-compatible"):
        self.name = f"{name}:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(f"{self.base_url}/chat/completions", headers=headers, timeout=TIMEOUT, json={
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class ClaudeLLM:
    def __init__(self, model: str, api_key: str):
        self.name = f"claude:{model}"
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str) -> str:
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=TIMEOUT, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
        }, json={
            "model": self.model, "max_tokens": 1024, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}],
        })
        r.raise_for_status()
        return "".join(block.get("text", "") for block in r.json()["content"])


def _key(env: str) -> str:
    key = os.getenv(env)
    if not key:
        raise RuntimeError(f"this backend needs your own key in {env}")
    return key


def make_llm(backend: str) -> LLM:
    model = os.getenv("RAGDOCS_LLM_MODEL")
    if backend == "claude":
        return ClaudeLLM(model or CLAUDE_MODEL, _key("ANTHROPIC_API_KEY"))
    if backend in PRESETS:
        p = PRESETS[backend]
        return OpenAICompatibleLLM(p["base_url"], model or p["model"],
                                   _key(p["key_env"]) if p["key_env"] else None, name=backend)
    raise ValueError(f"unknown LLM backend {backend!r}; choose one of {BACKENDS}")
