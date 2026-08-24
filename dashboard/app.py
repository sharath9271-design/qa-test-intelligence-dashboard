"""QA Test Intelligence Dashboard.

A Streamlit UI over the two halves of this project:

1. AI-assisted test-case generation (test_case_generator.py) with a
   human-in-the-loop review queue (review_store.py) - nothing an LLM
   drafts here is ever auto-approved.
2. Offline failure-message clustering (failure_clustering.py) - pure
   TF-IDF + k-means, no LLM involved, so it works identically with or
   without an API key.

Run locally with:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from failure_clustering import cluster_failures  # noqa: E402
from llm_client import AnthropicClient, FakeClient  # noqa: E402
from review_store import ReviewStore  # noqa: E402
from test_case_generator import generate_test_cases  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
# Overridable via env var so tests (and anyone running multiple independent
# dashboards) can point at an isolated store instead of the shared local one.
REVIEW_STORE_PATH = Path(os.environ.get("QA_DASHBOARD_REVIEW_STORE_PATH", DATA_DIR / "review_store.json"))

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_SPEC_PATH = FIXTURES_DIR / "sample_user_story.txt"
SAMPLE_FAILURES_PATH = FIXTURES_DIR / "sample_failure_messages.json"

DEMO_RESPONSE = """
### Test Case
Title: Sign in with valid credentials
Priority: High
Steps:
1. Navigate to the sign-in screen
2. Enter a registered email and its correct password
3. Click "Sign in"
Expected: The patient dashboard is shown with upcoming appointments

### Test Case
Title: Reject sign-in with an incorrect password
Priority: High
Steps:
1. Navigate to the sign-in screen
2. Enter a registered email and an incorrect password
3. Click "Sign in"
Expected: An "Invalid email or password." error is shown and the user stays on the sign-in screen

### Test Case
Title: Reject empty sign-in form client-side
Priority: Medium
Steps:
1. Navigate to the sign-in screen
2. Click "Sign in" without entering an email or password
Expected: An "Email and password are required." error is shown and no network request is sent

### Test Case
Title: Log out clears the session
Priority: Medium
Steps:
1. Sign in with valid credentials
2. Click "Log out"
Expected: The user is returned to the sign-in screen and a repeat visit to the dashboard redirects to sign-in
"""


def get_dashboard_client():
    """Real AnthropicClient when ANTHROPIC_API_KEY is set, otherwise a
    FakeClient seeded with a realistic canned response - so the
    generation flow is fully demoable offline, matching the "AI pieces
    are opt-in everywhere" pattern used across this portfolio."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return AnthropicClient(api_key=api_key), True
        except Exception:
            pass
    return FakeClient(canned_response=DEMO_RESPONSE), False


@st.cache_resource
def get_store(path: str) -> ReviewStore:
    return ReviewStore(path)


def load_sample_spec() -> str:
    if SAMPLE_SPEC_PATH.exists():
        return SAMPLE_SPEC_PATH.read_text()
    return ""


def load_sample_failures() -> list[str]:
    if SAMPLE_FAILURES_PATH.exists():
        return json.loads(SAMPLE_FAILURES_PATH.read_text())
    return []


def render_generate_tab() -> None:
    st.subheader("Generate test cases from a spec")
    st.caption(
        "Paste a user story or acceptance-criteria spec. The model drafts test "
        "cases in a fixed, parseable format; every draft lands in the Review "
        "Queue as 'pending' - nothing here is ever auto-approved."
    )

    client, is_real = get_dashboard_client()
    if is_real:
        st.success("Using a live Anthropic model (ANTHROPIC_API_KEY detected).", icon="✅")
    else:
        st.info(
            "No ANTHROPIC_API_KEY detected - using a bundled demo response so you "
            "can try the full flow end-to-end offline. Set ANTHROPIC_API_KEY to "
            "generate real test cases from your own spec.",
            icon="ℹ️",
        )

    if "spec_text" not in st.session_state:
        st.session_state.spec_text = ""

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Load sample spec"):
            st.session_state.spec_text = load_sample_spec()

    spec_text = st.text_area(
        "Specification",
        key="spec_text",
        height=220,
        placeholder="As a patient, I want to sign in to the portal so that I can view my appointments...",
    )

    if st.button("Generate test cases", type="primary", disabled=not spec_text.strip()):
        with st.spinner("Generating..."):
            try:
                cases = generate_test_cases(spec_text, client)
            except ValueError as exc:
                st.error(str(exc))
                cases = []

        if not cases:
            st.warning("No parseable test cases came back from the model.")
        else:
            store = get_store(str(REVIEW_STORE_PATH))
            added = 0
            for case in cases:
                case.id = f"tc-{uuid.uuid4().hex[:8]}"
                try:
                    store.add(case)
                    added += 1
                except ValueError:
                    continue
            st.success(f"Added {added} draft test case(s) to the Review Queue.")


def render_review_tab() -> None:
    st.subheader("Review queue")
    st.caption("Approve or reject each AI-drafted test case. Rejections and approvals can carry a note.")

    store = get_store(str(REVIEW_STORE_PATH))
    all_cases = store.list_all()

    counts = {status: len(store.list_by_status(status)) for status in ("pending", "approved", "rejected")}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(all_cases))
    m2.metric("Pending", counts["pending"])
    m3.metric("Approved", counts["approved"])
    m4.metric("Rejected", counts["rejected"])

    if not all_cases:
        st.info("No test cases yet - generate some from the 'Generate' tab first.")
        return

    status_filter = st.radio("Filter", ["pending", "approved", "rejected", "all"], horizontal=True, index=0)
    visible = all_cases if status_filter == "all" else store.list_by_status(status_filter)

    if not visible:
        st.info(f"No {status_filter} test cases.")
        return

    for case in visible:
        priority_badge = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(case.priority, "⚪")
        with st.expander(f"{priority_badge} [{case.priority}] {case.title}  ·  {case.status}"):
            st.markdown("**Steps:**")
            for i, step in enumerate(case.steps, start=1):
                st.markdown(f"{i}. {step}")
            st.markdown(f"**Expected:** {case.expected_result}")
            st.caption(f"id: {case.id}  ·  source: {case.source}")
            if case.review_note:
                st.caption(f"Review note: {case.review_note}")

            if case.status == "pending":
                note = st.text_input("Note (optional)", key=f"note-{case.id}")
                c1, c2 = st.columns(2)
                if c1.button("Approve", key=f"approve-{case.id}"):
                    store.approve(case.id, note=note)
                    st.rerun()
                if c2.button("Reject", key=f"reject-{case.id}"):
                    store.reject(case.id, note=note)
                    st.rerun()


def render_clustering_tab() -> None:
    st.subheader("Failure-message clustering")
    st.caption(
        "Groups similar CI failure messages via TF-IDF + k-means - fully offline, "
        "no model call. Turns a long list of one-off failures into a handful of "
        "recurring themes."
    )

    if "failure_text" not in st.session_state:
        st.session_state.failure_text = ""

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Load sample failures"):
            st.session_state.failure_text = "\n".join(load_sample_failures())

    uploaded = st.file_uploader("...or upload a JSON array / newline-delimited .txt of messages", type=["json", "txt"])
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                st.session_state.failure_text = "\n".join(str(m) for m in parsed)
            else:
                st.session_state.failure_text = raw
        except json.JSONDecodeError:
            st.session_state.failure_text = raw

    failure_text = st.text_area(
        "Failure messages (one per line)",
        key="failure_text",
        height=220,
    )

    auto_k = st.checkbox("Auto-pick number of clusters (silhouette score)", value=True)
    n_clusters = None
    if not auto_k:
        n_clusters = st.slider("Number of clusters", min_value=1, max_value=8, value=3)

    messages = [line.strip() for line in failure_text.splitlines() if line.strip()]

    if st.button("Cluster failures", type="primary", disabled=len(messages) < 2):
        with st.spinner("Clustering..."):
            result = cluster_failures(messages, n_clusters=n_clusters, random_state=0)

        st.success(f"Grouped {len(messages)} message(s) into {result.n_clusters} cluster(s).")

        summary_rows = [
            {"Cluster": c.id, "Size": c.size, "Top terms": ", ".join(c.top_terms)} for c in result.clusters
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        for cluster in sorted(result.clusters, key=lambda c: -c.size):
            with st.expander(f"Cluster {cluster.id} - {cluster.size} message(s) - {', '.join(cluster.top_terms)}"):
                for msg in cluster.messages:
                    st.markdown(f"- `{msg}`")
    elif len(messages) < 2:
        st.caption("Add at least two messages to cluster.")


def main() -> None:
    st.set_page_config(page_title="QA Test Intelligence Dashboard", layout="wide")
    st.title("QA Test Intelligence Dashboard")
    st.caption(
        "AI-assisted test-case drafting with human review, plus offline failure-message "
        "clustering - part of the [healthcare-qa-automation-framework] portfolio."
    )

    tab_generate, tab_review, tab_cluster = st.tabs(["Generate", "Review Queue", "Failure Clustering"])
    with tab_generate:
        render_generate_tab()
    with tab_review:
        render_review_tab()
    with tab_cluster:
        render_clustering_tab()


if __name__ == "__main__":
    main()
