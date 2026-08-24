"""A tiny JSON-file-backed store for human-in-the-loop review of
AI-generated test cases.

Every case generate_test_cases() produces starts as "pending"; nothing
here ever auto-approves a case. This store is deliberately simple (a
JSON file, not a database) - the point is to make the review workflow
itself (add -> list pending -> approve/reject with a note) concrete and
testable, not to be a production persistence layer.
"""
from __future__ import annotations

import json
from pathlib import Path

from test_case_generator import TestCase


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def _save(self, cases: list[dict]) -> None:
        self.path.write_text(json.dumps(cases, indent=2))

    def add(self, case: TestCase) -> None:
        cases = self._load()
        if any(c["id"] == case.id for c in cases):
            raise ValueError(f"ReviewStore: a case with id {case.id!r} already exists.")
        cases.append(case.to_dict())
        self._save(cases)

    def add_many(self, cases: list[TestCase]) -> None:
        for case in cases:
            self.add(case)

    def get(self, case_id: str) -> TestCase:
        for c in self._load():
            if c["id"] == case_id:
                return TestCase.from_dict(c)
        raise KeyError(f"ReviewStore: no case with id {case_id!r}.")

    def list_all(self) -> list[TestCase]:
        return [TestCase.from_dict(c) for c in self._load()]

    def list_by_status(self, status: str) -> list[TestCase]:
        return [c for c in self.list_all() if c.status == status]

    def approve(self, case_id: str, note: str = "") -> None:
        self._set_status(case_id, "approved", note)

    def reject(self, case_id: str, note: str = "") -> None:
        self._set_status(case_id, "rejected", note)

    def _set_status(self, case_id: str, status: str, note: str) -> None:
        cases = self._load()
        for c in cases:
            if c["id"] == case_id:
                c["status"] = status
                c["review_note"] = note
                self._save(cases)
                return
        raise KeyError(f"ReviewStore: no case with id {case_id!r}.")
