from __future__ import annotations

import pytest

from review_store import ReviewStore
from test_case_generator import TestCase


def make_case(case_id: str = "tc-1") -> TestCase:
    return TestCase(
        id=case_id,
        title="Sign in with valid credentials",
        steps=["Go to sign-in", "Enter credentials", "Submit"],
        expected_result="Dashboard is shown",
        priority="High",
    )


def test_new_store_starts_empty(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    assert store.list_all() == []


def test_add_and_list_all(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add(make_case())

    all_cases = store.list_all()
    assert len(all_cases) == 1
    assert all_cases[0].id == "tc-1"
    assert all_cases[0].status == "pending"


def test_add_rejects_duplicate_ids(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add(make_case("tc-1"))
    with pytest.raises(ValueError, match="already exists"):
        store.add(make_case("tc-1"))


def test_add_many(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add_many([make_case("tc-1"), make_case("tc-2")])
    assert len(store.list_all()) == 2


def test_approve_sets_status_and_note(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add(make_case())
    store.approve("tc-1", note="Looks correct, matches the acceptance criteria.")

    case = store.get("tc-1")
    assert case.status == "approved"
    assert case.review_note == "Looks correct, matches the acceptance criteria."


def test_reject_sets_status_and_note(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add(make_case())
    store.reject("tc-1", note="Missing the negative case from the spec.")

    case = store.get("tc-1")
    assert case.status == "rejected"
    assert case.review_note == "Missing the negative case from the spec."


def test_list_by_status_filters_correctly(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    store.add_many([make_case("tc-1"), make_case("tc-2"), make_case("tc-3")])
    store.approve("tc-1")
    store.reject("tc-2")

    assert [c.id for c in store.list_by_status("pending")] == ["tc-3"]
    assert [c.id for c in store.list_by_status("approved")] == ["tc-1"]
    assert [c.id for c in store.list_by_status("rejected")] == ["tc-2"]


def test_get_raises_for_unknown_id(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_approve_raises_for_unknown_id(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    with pytest.raises(KeyError):
        store.approve("does-not-exist")


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "review.json"
    ReviewStore(path).add(make_case())

    # A fresh ReviewStore instance pointed at the same file should see the
    # same data - this is the whole point of persisting to disk rather
    # than keeping state only in memory.
    reopened = ReviewStore(path)
    assert len(reopened.list_all()) == 1
    assert reopened.list_all()[0].id == "tc-1"
