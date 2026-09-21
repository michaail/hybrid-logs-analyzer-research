"""Simple non-graph baselines under the same frozen graph split contract."""

from __future__ import annotations

from typing import Any

import numpy as np


def window_feature_matrices(
    train_graphs: list[Any],
    *splits: list[Any],
    oov_cluster_id: int = -1,
) -> tuple[np.ndarray, ...]:
    """Build train-vocabulary count features for BGL windows.

    Every row contains counts for template IDs observed in the training split,
    plus the OOV-event share and ``log1p`` window length. Unknown non-OOV
    templates in validation/test are deliberately not added to the vocabulary.
    """
    vocabulary = sorted(
        {
            int(cluster_id)
            for graph in train_graphs
            for cluster_id in _cluster_ids(graph)
            if int(cluster_id) >= 0
        }
    )
    if not vocabulary:
        raise ValueError("Isolation Forest baseline found no known training templates.")
    index = {cluster_id: position for position, cluster_id in enumerate(vocabulary)}

    def transform(graphs: list[Any]) -> np.ndarray:
        matrix = np.zeros((len(graphs), len(vocabulary) + 2), dtype=np.float32)
        for row, graph in enumerate(graphs):
            cluster_ids = _cluster_ids(graph)
            n_events = len(cluster_ids)
            if not n_events:
                continue
            for cluster_id in cluster_ids:
                if cluster_id in index:
                    matrix[row, index[cluster_id]] += 1.0
            matrix[row, -2] = sum(cluster_id == oov_cluster_id for cluster_id in cluster_ids) / n_events
            matrix[row, -1] = float(np.log1p(n_events))
        return matrix

    return tuple(transform(graphs) for graphs in (train_graphs, *splits))


def _cluster_ids(graph: Any) -> list[int]:
    """Return original event-level cluster IDs retained in a PyG graph."""
    value = getattr(graph, "event_cluster_ids", None)
    if value is None:
        raise ValueError(
            "Graph bundle lacks event_cluster_ids required by Isolation Forest. "
            "Rebuild the BGL campaign graphs with the current pipeline."
        )
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [int(cluster_id) for cluster_id in value]