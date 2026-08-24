from __future__ import annotations

import pytest

from llm_client import FakeClient
from test_case_generator import TestCase, build_prompt, generate_test_cases, parse_test_cases

SAMPLE_RESPONSE = """
### Test Case
Title: Sign in with valid credentials
Priority: High
Steps:
1. Navigate to the sign-in screen
2. Enter a valid email and password
3. Click "Sign in"
Expected: The dashboard is shown with a welcome message and upcoming appointments

### Test Case
Title: Reject sign-in with an incorrect password
Priority: High
Steps:
1. Navigate to the sign-in screen
2. Enter a valid email and an incorrect password
3. Click "Sign in"
Expected: An "Invalid email or password." error is shown and the user stays on the sign-in screen

### Test Case
Title: Reject empty sign-in form client-side
Priority: Medium
Steps:
1. Navigate to the sign-in screen
2. Click "Sign in" without entering anything
Expected: An "Email and password are required." error is shown and no request is sent
"""


def test_parse_test_cases_extracts_every_block():
    cases = parse_test_cases(SAMPLE_RESPONSE)
    assert len(cases) == 3
    assert [c.id for c in cases] == ["tc-1", "tc-2", "tc-3"]


def test_parse_test_cases_extracts_fields_correctly():
    cases = parse_test_cases(SAMPLE_RESPONSE)
    first = cases[0]

    assert first.title == "Sign in with valid credentials"
    assert first.priority == "High"
    assert first.steps == [
        "Navigate to the sign-in screen",
        "Enter a valid email and password",
        'Click "Sign in"',
    ]
    assert "dashboard is shown" in first.expected_result
    assert first.status == "pending"
    assert first.source == "llm"


def test_parse_test_cases_respects_source_override():
    cases = parse_test_cases(SAMPLE_RESPONSE, source="heuristic")
    assert all(c.source == "heuristic" for c in cases)


def test_parse_test_cases_defaults_invalid_priority_to_medium():
    raw = """
    ### Test Case
    Title: Something
    Priority: Critical
    Steps:
    1. Do a thing
    Expected: A result
    """
    cases = parse_test_cases(raw)
    assert cases[0].priority == "Medium"


def test_parse_test_cases_skips_block_with_no_title():
    raw = """
    ### Test Case
    Priority: High
    Steps:
    1. Do a thing
    Expected: A result
    """
    assert parse_test_cases(raw) == []


def test_parse_test_cases_skips_block_with_no_steps():
    raw = """
    ### Test Case
    Title: A title with no steps
    Priority: High
    Expected: A result
    """
    assert parse_test_cases(raw) == []


def test_parse_test_cases_handles_empty_input():
    assert parse_test_cases("") == []
    assert parse_test_cases("no structured content here") == []


def test_build_prompt_includes_the_spec_verbatim():
    prompt = build_prompt("As a user I want to log in.")
    assert "As a user I want to log in." in prompt
    assert "Title:" in prompt
    assert "Steps:" in prompt


def test_generate_test_cases_calls_the_client_and_parses_the_result():
    client = FakeClient(canned_response=SAMPLE_RESPONSE)
    cases = generate_test_cases("As a returning patient, I want to sign in.", client)

    assert len(cases) == 3
    assert len(client.prompts_seen) == 1
    assert "As a returning patient, I want to sign in." in client.prompts_seen[0]


def test_generate_test_cases_rejects_empty_spec():
    client = FakeClient(canned_response=SAMPLE_RESPONSE)
    with pytest.raises(ValueError, match="must not be empty"):
        generate_test_cases("   ", client)


def test_generate_test_cases_returns_empty_list_for_unparseable_response():
    client = FakeClient(canned_response="I cannot help with that.")
    cases = generate_test_cases("A spec.", client)
    assert cases == []


def test_testcase_round_trips_through_dict():
    case = TestCase(id="tc-1", title="T", steps=["a", "b"], expected_result="E", priority="Low")
    restored = TestCase.from_dict(case.to_dict())
    assert restored == case
