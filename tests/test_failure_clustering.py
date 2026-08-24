from __future__ import annotations

import json
import pathlib

from failure_clustering import cluster_failures

FIXTURES_PATH = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "sample_failure_messages.json"


def load_sample_messages() -> list[str]:
    return json.loads(FIXTURES_PATH.read_text())


def test_clusters_the_sample_failures_into_three_known_themes():
    messages = load_sample_messages()
    result = cluster_failures(messages, n_clusters=3, random_state=0)

    assert result.n_clusters == 3
    assert sum(c.size for c in result.clusters) == len(set(m.strip() for m in messages))

    # Every message should land in exactly one cluster.
    all_clustered = [m for c in result.clusters for m in c.messages]
    assert sorted(all_clustered) == sorted(set(messages))


def test_timeout_messages_cluster_together():
    messages = load_sample_messages()
    result = cluster_failures(messages, n_clusters=3, random_state=0)

    timeout_messages = [m for m in messages if "Timeout" in m]
    cluster_ids = {
        cluster.id for cluster in result.clusters for m in cluster.messages if m in timeout_messages
    }
    # All timeout-related messages should be in the same cluster as each other.
    assert len(cluster_ids) == 1


def test_network_error_messages_cluster_separately_from_timeouts():
    messages = load_sample_messages()
    result = cluster_failures(messages, n_clusters=3, random_state=0)

    def cluster_for(substr: str) -> int:
        for cluster in result.clusters:
            if any(substr in m for m in cluster.messages):
                return cluster.id
        raise AssertionError(f"no cluster contains a message with {substr!r}")

    assert cluster_for("Timeout") != cluster_for("hapi.fhir.org")


def test_top_terms_are_populated_and_relevant():
    messages = load_sample_messages()
    result = cluster_failures(messages, n_clusters=3, random_state=0)

    for cluster in result.clusters:
        assert len(cluster.top_terms) > 0
        assert all(isinstance(term, str) and term for term in cluster.top_terms)


def test_deduplicates_identical_messages():
    messages = ["same error", "same error", "same error", "a totally different error"]
    result = cluster_failures(messages, n_clusters=2, random_state=0)

    total_messages = sum(c.size for c in result.clusters)
    assert total_messages == 2  # deduplicated


def test_handles_empty_input():
    result = cluster_failures([], random_state=0)
    assert result.clusters == []
    assert result.n_clusters == 0


def test_handles_a_single_message():
    result = cluster_failures(["only one message"], random_state=0)
    assert result.n_clusters == 1
    assert result.clusters[0].messages == ["only one message"]


def test_auto_picks_k_when_not_specified():
    messages = load_sample_messages()
    result = cluster_failures(messages, random_state=0)
    # The auto-picked k should be reasonable for 3 obviously distinct themes.
    assert 2 <= result.n_clusters <= 6


def test_clustering_is_deterministic_given_a_fixed_random_state():
    messages = load_sample_messages()
    result_a = cluster_failures(messages, n_clusters=3, random_state=42)
    result_b = cluster_failures(messages, n_clusters=3, random_state=42)

    a_by_cluster = sorted(tuple(sorted(c.messages)) for c in result_a.clusters)
    b_by_cluster = sorted(tuple(sorted(c.messages)) for c in result_b.clusters)
    assert a_by_cluster == b_by_cluster
