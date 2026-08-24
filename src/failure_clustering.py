"""Groups similar CI failure messages together via TF-IDF + k-means, so a
long list of individually-triaged failures (see triage_failure.py in the
healthcare-qa-automation-framework repo) can be read as a handful of
recurring *themes* instead of N unrelated one-off errors.

Fully offline - no LLM, no network call - which is what makes this the
"quality analytics" half of this project rather than the AI-generation
half (see test_case_generator.py for that).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


@dataclass
class Cluster:
    id: int
    top_terms: list[str]
    messages: list[str]

    @property
    def size(self) -> int:
        return len(self.messages)


@dataclass
class ClusteringResult:
    clusters: list[Cluster]
    n_clusters: int


def cluster_failures(
    messages: list[str],
    *,
    n_clusters: int | None = None,
    top_terms_per_cluster: int = 5,
    random_state: int = 0,
) -> ClusteringResult:
    """Clusters `messages` by TF-IDF cosine similarity via k-means.

    If n_clusters is None, sweeps k from 2 to min(6, n-1) and picks the
    best silhouette score - useful for an interactive dashboard. Pass an
    explicit n_clusters for deterministic, fast tests.
    """
    unique = list(dict.fromkeys(m.strip() for m in messages if m.strip()))
    if len(unique) < 2:
        single = [Cluster(id=0, top_terms=[], messages=unique)] if unique else []
        return ClusteringResult(clusters=single, n_clusters=len(single))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(unique)
    terms = np.array(vectorizer.get_feature_names_out())

    k = n_clusters if n_clusters is not None else _pick_k(matrix, len(unique), random_state)
    k = max(1, min(k, len(unique)))

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    clusters: list[Cluster] = []
    for cluster_id in range(k):
        member_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if not member_indices:
            continue
        centroid = model.cluster_centers_[cluster_id]
        top_indices = centroid.argsort()[::-1][:top_terms_per_cluster]
        clusters.append(
            Cluster(
                id=cluster_id,
                top_terms=[t for t in terms[top_indices] if t],
                messages=[unique[i] for i in member_indices],
            )
        )
    return ClusteringResult(clusters=clusters, n_clusters=len(clusters))


def _pick_k(matrix, n_samples: int, random_state: int) -> int:
    max_k = min(6, n_samples - 1)
    if max_k < 2:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k
