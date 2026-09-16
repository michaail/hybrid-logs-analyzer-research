"""Stage 4 — Collapsed-template graph builder.

Converts a per-sequence DataFrame into a NetworkX DiGraph using the
**collapsed-template** representation:

* One node per unique ``cluster_id`` in the sequence.
* Node features: occurrence count, parameter statistics, and 5 positional dims.
* Edge features: transition count, 7 time-delta distribution dims, and 3
  positional dims (10 dims total).
* When ``use_edge_features=False`` edges carry only the transition count
  (``weight``), reducing edge dimensionality for ablation experiments.

Also provides sequence fingerprinting for deduplication: sequences sharing
the same ordered ``cluster_id`` list are structurally identical and can
reuse a single canonical graph for efficiency.

Public API
----------
    build_collapsed_graph(seq, cluster_to_template, *, use_edge_features) → nx.DiGraph
    sequence_fingerprint(seq)                                               → str
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from typing import Any, cast

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def build_collapsed_graph(
    seq: pd.DataFrame,
    cluster_to_template: dict,
    *,
    use_edge_features: bool = True,
) -> nx.DiGraph:
    """Build a collapsed-template directed graph from a sequence DataFrame.

    Parameters
    ----------
    seq : pd.DataFrame
        Single-sequence DataFrame (one block or one window).  Must have
        ``cluster_id`` and ``parameters`` columns plus either ``timestamp``
        (HDFS datetime objects) or ``unix_ts`` (BGL integer seconds).
    cluster_to_template : dict
        Mapping ``cluster_id → template string``.
    use_edge_features : bool
        When ``False``, edges carry only ``weight`` (transition count).
        All time-delta and positional edge features are omitted.

    Returns
    -------
    nx.DiGraph
        Node attributes::

            template, occurrence_count, param_count, param_num_mean,
            param_num_max, first_pos, last_pos, mean_pos, std_pos, pos_spread

        Edge attributes (when ``use_edge_features=True``)::

            weight, mean_src_pos, mean_dst_pos, mean_pos_delta,
            td_min, td_p25, td_median, td_p75, td_max, td_std
    """
    G = nx.DiGraph()

    # Store sequence identifier for traceability
    for id_col in ("window_id", "block_id"):
        if id_col in seq.columns:
            G.graph[id_col] = seq[id_col].iloc[0]
            break

    cids = seq["cluster_id"].tolist()
    params = seq["parameters"].tolist()
    ts = _extract_timestamps(seq)
    n = len(cids)

    # ── Per-node aggregation ──────────────────────────────────────────────────
    node_params: dict = defaultdict(list)
    node_count = Counter(cids)
    node_positions: dict = defaultdict(list)

    for i, (cid, p) in enumerate(zip(cids, params)):
        node_params[cid].extend(p)
        node_positions[cid].append(i / max(n - 1, 1))

    for cid, count in node_count.items():
        nums = [float(x) for x in node_params[cid] if _is_numeric(x)]
        positions = np.array(node_positions[cid])

        G.add_node(
            cid,
            template=cluster_to_template.get(cid, ""),
            occurrence_count=count,
            param_count=len(node_params[cid]),
            param_num_mean=float(np.mean(nums)) if nums else 0.0,
            param_num_max=float(np.max(nums)) if nums else 0.0,
            first_pos=float(positions.min()),
            last_pos=float(positions.max()),
            mean_pos=float(positions.mean()),
            std_pos=float(positions.std()) if len(positions) > 1 else 0.0,
            pos_spread=float(positions.max() - positions.min()),
        )

    # ── Per-edge aggregation ──────────────────────────────────────────────────
    edge_deltas: dict = defaultdict(list)
    edge_src_pos: dict = defaultdict(list)
    edge_dst_pos: dict = defaultdict(list)

    for i in range(n - 1):
        src, dst = cids[i], cids[i + 1]
        src_norm = i / max(n - 1, 1)
        dst_norm = (i + 1) / max(n - 1, 1)

        edge_src_pos[(src, dst)].append(src_norm)
        edge_dst_pos[(src, dst)].append(dst_norm)

        t_src, t_dst = ts[i], ts[i + 1]
        if t_src is not None and t_dst is not None:
            try:
                delta = t_dst - t_src
                if hasattr(delta, "total_seconds"):
                    delta = delta.total_seconds()
                edge_deltas[(src, dst)].append(float(delta))
            except TypeError:
                if (src, dst) not in edge_deltas:
                    edge_deltas[(src, dst)]
        elif (src, dst) not in edge_deltas:
            edge_deltas[(src, dst)]

    for (src, dst) in edge_src_pos:
        s_pos = np.array(edge_src_pos[(src, dst)])
        d_pos = np.array(edge_dst_pos[(src, dst)])
        deltas = edge_deltas.get((src, dst), [])

        edge_attrs: dict = {"weight": len(s_pos)}

        if use_edge_features:
            edge_attrs.update(
                mean_src_pos=float(s_pos.mean()),
                mean_dst_pos=float(d_pos.mean()),
                mean_pos_delta=float((d_pos - s_pos).mean()),
            )
            if deltas:
                arr = np.array(deltas)
                edge_attrs.update(
                    td_min=float(arr.min()),
                    td_p25=float(np.percentile(arr, 25)),
                    td_median=float(np.median(arr)),
                    td_p75=float(np.percentile(arr, 75)),
                    td_max=float(arr.max()),
                    td_std=float(arr.std()),
                )
            else:
                edge_attrs.update(
                    td_min=-1, td_p25=-1, td_median=-1,
                    td_p75=-1, td_max=-1, td_std=0,
                )

        G.add_edge(src, dst, **edge_attrs)

    return G


def sequence_fingerprint(seq: pd.DataFrame) -> str:
    """Hash the ordered ``cluster_id`` sequence to a compact hex fingerprint.

    Two sequences with the same fingerprint have identical template orderings
    (same structural graph topology), though timestamps and parameters may differ.
    """
    cids = seq["cluster_id"].tolist()
    key = "|".join(str(c) for c in cids)
    return hashlib.md5(key.encode()).hexdigest()  # noqa: S324 — not security-sensitive


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_timestamps(seq: pd.DataFrame) -> list:
    """Return a list of timestamps compatible with delta computation.

    Prefers ``unix_ts`` (numeric seconds) over ``timestamp`` (datetime).
    """
    if "unix_ts" in seq.columns:
        return [float(t) if t is not None else None for t in seq["unix_ts"].tolist()]
    if "timestamp" in seq.columns:
        return seq["timestamp"].tolist()
    return [None] * len(seq)


def _is_numeric(x: object) -> bool:
    """Return True only for finite non-NaN non-Inf numeric values."""
    try:
        v = float(cast(Any, x))
        return np.isfinite(v)
    except (ValueError, TypeError):
        return False
