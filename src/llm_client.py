"""Pluggable LLM client for AI-assisted test-case generation.

Same dependency-injection pattern used across this portfolio
(healing/selector_suggester.ts and scripts/llm_client.py in
healthcare-qa-automation-framework, github/githubClient.ts in
qa-copilot-mcp-server): an interface, a real client gated behind an API
key, and a deterministic fake used in tests. That keeps the parsing logic
in test_case_generator.py fully unit-testable with zero external calls,
while the real integration is genuine, runnable code.
"""
from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return a completion for `prompt`."""
        ...


class FakeClient:
    """Deterministic stand-in used in tests and whenever no API key is
    configured. Records every prompt it was asked to complete."""

    def __init__(self, canned_response: str | None = None):
        self.canned_response = canned_response
        self.prompts_seen: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        if self.canned_response is not None:
            return self.canned_response
        return ""


class AnthropicClient:
    """Real client, gated entirely behind ANTHROPIC_API_KEY. The SDK is
    imported lazily so it is never required just to run this repo's
    tests."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("AnthropicClient requires ANTHROPIC_API_KEY to be set.")

    def complete(self, prompt: str) -> str:
        import anthropic  # noqa: PLC0415 - deliberately lazy, see class docstring

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        return "\n".join(parts).strip()


def get_client() -> LLMClient:
    """Factory: a real AnthropicClient if ANTHROPIC_API_KEY is set and the
    SDK is available, a FakeClient otherwise. Never raises."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except Exception:
            return FakeClient()
    return FakeClient()
