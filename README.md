# QA Test Intelligence Dashboard

A Streamlit dashboard for two things QA teams do constantly and usually do
by hand: drafting test cases from a spec, and making sense of a pile of CI
failure messages.

- **AI-assisted test-case generation**, with a human-in-the-loop review
  queue — the model drafts, a person approves, edits are always possible,
  nothing ships un-reviewed.
- **Offline failure-message clustering** — TF-IDF + k-means groups a long
  list of individually-triaged CI failures into a handful of recurring
  themes, entirely without an LLM.

This is the fifth project in a small portfolio of real, CI-verified QA
automation tooling; see [`healthcare-qa-automation-framework`](https://github.com/sharath9271-design/healthcare-qa-automation-framework),
[`healthcare-data-quality-framework`](https://github.com/sharath9271-design/healthcare-data-quality-framework),
and [`qa-copilot-mcp-server`](https://github.com/sharath9271-design/qa-copilot-mcp-server)
for the others.

## Screenshots-in-words

- **Generate tab**: paste a spec (or load the bundled sample), click
  *Generate test cases*, and structured drafts (title, priority, steps,
  expected result) land in the review queue as `pending`.
- **Review Queue tab**: approve or reject each draft, optionally with a
  note. Status counts are always visible at the top.
- **Failure Clustering tab**: paste or upload failure messages (or load the
  bundled sample), pick auto-k or a fixed cluster count, and see each
  cluster's size and top TF-IDF terms.

## Architecture

```
qa-test-intelligence-dashboard/
├── src/
│   ├── llm_client.py          # LLMClient protocol + FakeClient + AnthropicClient
│   ├── test_case_generator.py # prompt building + deterministic response parsing
│   ├── review_store.py        # JSON-file-backed human-in-the-loop review queue
│   └── failure_clustering.py  # TF-IDF + k-means clustering, zero LLM involved
├── dashboard/
│   └── app.py                 # Streamlit UI tying the three modules together
├── fixtures/                  # sample spec + sample failure messages, used by
│                               # both the test suite and the dashboard's "load sample" buttons
├── tests/                     # 35 tests: unit tests for src/, plus dashboard
│                               # smoke tests driven through streamlit.testing.v1
└── .github/workflows/ci.yml
```

## Why it's built this way

**The pluggable-LLM pattern, a third time.** `llm_client.py` reuses the
same `Protocol` / `FakeClient` / real-client shape as the Python triage
script in `healthcare-qa-automation-framework` and the `GitHubClient`
interface in `qa-copilot-mcp-server`. `AnthropicClient` imports the
`anthropic` SDK lazily, inside `complete()`, so it's never a hard
dependency — the whole test suite, and the dashboard's demo mode, run with
`anthropic` not even installed.

**The LLM's job is narrow and structured, so it's fully testable without
one.** `test_case_generator.py` asks the model for a fixed, parseable
format (`### Test Case` / `Title:` / `Priority:` / `Steps:` / `Expected:`)
and does all the actual logic — parsing, id assignment, validation — in
plain, deterministic Python. `tests/test_test_case_generator.py` exercises
every parsing edge case (missing title, missing steps, invalid priority,
unparseable response) against a fixed `FakeClient` response. Zero real
model calls in CI.

**Nothing an LLM drafts is ever auto-approved.** Every generated
`TestCase` starts as `status="pending"`. `review_store.py` is a small
JSON-file-backed store whose only job is to make "add → list pending →
approve/reject with a note" concrete, inspectable, and testable — not to
be a production database.

**The analytics half needs no mocking at all.** `failure_clustering.py` is
plain scikit-learn: `TfidfVectorizer` + `KMeans`, with an optional
silhouette-score sweep to auto-pick `k` when the caller doesn't specify
one. `tests/test_failure_clustering.py` runs it for real against a fixture
of 12 realistic failure messages across 3 themes (timeouts, strict-mode
locator violations, FHIR API/network errors) and asserts the clustering
actually separates them correctly — this is genuine unsupervised learning,
verified against real behavior, not a mocked-out placeholder.

**The dashboard is tested as a real app, not just imported.** `dashboard/
app.py` runs under `streamlit.testing.v1.AppTest` in
`tests/test_dashboard_smoke.py`, which clicks through both flows for
real: load the sample spec → generate → verify the review queue populates
→ approve a case → verify counts update, and load sample failures →
cluster → verify a results table renders. Each test points
`QA_DASHBOARD_REVIEW_STORE_PATH` at an isolated temp file so tests never
share state or touch the local `data/` directory.

## Running it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the test suite
python -m pytest -v

# launch the dashboard
streamlit run dashboard/app.py
```

Without `ANTHROPIC_API_KEY` set, the Generate tab uses a bundled demo
response so the full flow (generate → review → approve/reject) works
completely offline. Set `ANTHROPIC_API_KEY` (and `pip install anthropic`,
which is deliberately not in `requirements.txt` — see the comment there)
to generate real test cases from your own spec text.

Review-queue state is stored in `data/review_store.json`, created on first
use and gitignored.

## Tech stack

Python 3.11, Streamlit, scikit-learn, pandas, numpy, pytest,
`streamlit.testing.v1` for UI-level smoke tests.

## CI

GitHub Actions installs the pinned `requirements.txt` into a clean
environment and runs the full `pytest` suite — unit tests for the
generation, review-store, and clustering modules, plus the dashboard smoke
tests that drive the real Streamlit app end-to-end. No network access and
no API key are required for CI to go green.
