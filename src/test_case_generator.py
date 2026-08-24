"""AI-assisted test-case generation from a spec or user story.

The LLM's job here is narrow and structured: given a spec, respond with a
fixed, parseable format (one block per test case: Title / Priority / Steps
/ Expected). All the actual logic that matters - parsing that format into
real objects, assigning ids, and the human-in-the-loop review workflow
(see review_store.py) - is deterministic and lives in plain Python, so it
is fully unit-testable against a fixed FakeClient response (see
tests/test_test_case_generator.py) without ever calling a real model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import LLMClient

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_STATUSES = {"pending", "approved", "rejected"}


@dataclass
class TestCase:
    id: str
    title: str
    steps: list[str]
    expected_result: str
    priority: str = "Medium"
    source: str = "llm"
    status: str = "pending"
    review_note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "steps": self.steps,
            "expected_result": self.expected_result,
            "priority": self.priority,
            "source": self.source,
            "status": self.status,
            "review_note": self.review_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(
            id=data["id"],
            title=data["title"],
            steps=list(data["steps"]),
            expected_result=data["expected_result"],
            priority=data.get("priority", "Medium"),
            source=data.get("source", "llm"),
            status=data.get("status", "pending"),
            review_note=data.get("review_note", ""),
        )


def build_prompt(spec_text: str) -> str:
    return (
        "You are a QA engineer writing test cases from a specification.\n"
        "For each distinct behavior worth testing, output a block in EXACTLY this format "
        "(no extra commentary, no markdown headers other than shown):\n\n"
        "### Test Case\n"
        "Title: <short imperative title>\n"
        "Priority: <High|Medium|Low>\n"
        "Steps:\n"
        "1. <step>\n"
        "2. <step>\n"
        "Expected: <expected result>\n\n"
        "Repeat the block for every test case. Cover both the happy path and at least one "
        "negative/edge case where the spec implies one.\n\n"
        f"Specification:\n{spec_text}\n"
    )


_BLOCK_RE = re.compile(r"###\s*Test Case.*?(?=###\s*Test Case|\Z)", re.DOTALL)
_TITLE_RE = re.compile(r"Title:\s*(.+)")
_PRIORITY_RE = re.compile(r"Priority:\s*(\w+)")
_STEP_RE = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
_EXPECTED_RE = re.compile(r"Expected:\s*(.+)")


def parse_test_cases(raw_text: str, *, source: str = "llm") -> list[TestCase]:
    """Parses the structured LLM response format into TestCase objects.

    Malformed or incomplete blocks (missing a title, or with zero steps)
    are skipped rather than raised on - a partially-broken response should
    still yield whatever test cases it validly contains, since a human
    reviews every generated case before it's trusted anyway.
    """
    cases: list[TestCase] = []
    for i, block in enumerate(_BLOCK_RE.findall(raw_text), start=1):
        title_match = _TITLE_RE.search(block)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        if not title:
            continue

        steps = [s.strip() for s in _STEP_RE.findall(block) if s.strip()]
        if not steps:
            continue

        priority_match = _PRIORITY_RE.search(block)
        priority = priority_match.group(1).strip() if priority_match else "Medium"
        if priority not in VALID_PRIORITIES:
            priority = "Medium"

        expected_match = _EXPECTED_RE.search(block)
        expected_result = expected_match.group(1).strip() if expected_match else ""

        cases.append(
            TestCase(
                id=f"tc-{i}",
                title=title,
                steps=steps,
                expected_result=expected_result,
                priority=priority,
                source=source,
                status="pending",
            )
        )
    return cases


def generate_test_cases(spec_text: str, client: LLMClient, *, source: str = "llm") -> list[TestCase]:
    """Calls the LLM client with a structured prompt and parses the result.

    Every generated case starts life with status="pending" - it's a draft
    for a human to approve, edit, or reject via review_store.py, not a
    finished artifact. That review step is the point: this function
    accelerates drafting, it doesn't replace judgment.
    """
    if not spec_text.strip():
        raise ValueError("generate_test_cases: spec_text must not be empty.")

    raw = client.complete(build_prompt(spec_text))
    return parse_test_cases(raw, source=source)
