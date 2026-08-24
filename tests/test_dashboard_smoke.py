"""End-to-end smoke tests for the Streamlit dashboard, using Streamlit's
own AppTest harness (streamlit.testing.v1) rather than mocking Streamlit
away. These exercise the real script: widget interactions, session state,
and the underlying src/ modules together - not just that it imports.

Each test points QA_DASHBOARD_REVIEW_STORE_PATH at a fresh tmp_path file so
runs never touch (or depend on) the repo's local data/ directory, and tests
never see each other's review-store state.
"""
from __future__ import annotations

import pathlib

from streamlit.testing.v1 import AppTest

APP_PATH = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def make_app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setenv("QA_DASHBOARD_REVIEW_STORE_PATH", str(tmp_path / "review_store.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return AppTest.from_file(str(APP_PATH))


def test_dashboard_loads_without_exceptions(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.run(timeout=30)

    assert not at.exception
    assert len(at.tabs) == 3
    assert at.title[0].value == "QA Test Intelligence Dashboard"


def test_generating_from_the_sample_spec_populates_the_review_queue(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.run(timeout=30)

    generate_tab = at.tabs[0]
    [b for b in generate_tab.button if b.label == "Load sample spec"][0].click().run(timeout=30)
    assert not at.exception

    generate_tab = at.tabs[0]
    generate_button = [b for b in generate_tab.button if b.label == "Generate test cases"][0]
    assert generate_button.disabled is False
    generate_button.click().run(timeout=30)
    assert not at.exception

    review_tab = at.tabs[1]
    metrics = {m.label: m.value for m in review_tab.metric}
    # The bundled demo response (used when no ANTHROPIC_API_KEY is set)
    # yields exactly 4 parseable test cases, all starting as pending.
    assert metrics["Total"] == "4"
    assert metrics["Pending"] == "4"
    assert metrics["Approved"] == "0"


def test_approving_a_case_moves_it_out_of_pending(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.run(timeout=30)

    generate_tab = at.tabs[0]
    [b for b in generate_tab.button if b.label == "Load sample spec"][0].click().run(timeout=30)
    generate_tab = at.tabs[0]
    [b for b in generate_tab.button if b.label == "Generate test cases"][0].click().run(timeout=30)

    review_tab = at.tabs[1]
    approve_buttons = [b for b in review_tab.button if b.label == "Approve"]
    assert len(approve_buttons) == 4
    approve_buttons[0].click().run(timeout=30)
    assert not at.exception

    review_tab = at.tabs[1]
    metrics = {m.label: m.value for m in review_tab.metric}
    assert metrics["Pending"] == "3"
    assert metrics["Approved"] == "1"


def test_clustering_the_sample_failures_renders_a_results_table(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.run(timeout=30)

    cluster_tab = at.tabs[2]
    [b for b in cluster_tab.button if b.label == "Load sample failures"][0].click().run(timeout=30)
    assert not at.exception

    cluster_tab = at.tabs[2]
    cluster_button = [b for b in cluster_tab.button if b.label == "Cluster failures"][0]
    assert cluster_button.disabled is False
    cluster_button.click().run(timeout=30)
    assert not at.exception

    cluster_tab = at.tabs[2]
    assert len(cluster_tab.dataframe) == 1
