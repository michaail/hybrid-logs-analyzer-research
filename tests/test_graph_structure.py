from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modules.dataset import (
    NODE_EXTRA_DIM,
    STRUCTURE_EDGE_DIM,
    _node_features_from_structure,
    _seq_to_structure,
    build_graph_structures,
    load_graph_structures,
    save_graph_structures,
)


def _hdfs_frame(block_id: str, cluster_ids: list[int], *, seconds: list[int] | None = None) -> pd.DataFrame:
    n = len(cluster_ids)
    offsets = seconds if seconds is not None else list(range(n))
    return pd.DataFrame(
        {
            "block_id": [block_id] * n,
            "cluster_id": cluster_ids,
            "parameters": [[float(i)] for i in range(n)],
            "timestamp": [pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=s) for s in offsets],
        }
    )


def test_structure_is_embedding_agnostic_and_always_stores_full_edges() -> None:
    seq = _hdfs_frame("blk_a", [1, 2, 1])
    structure = _seq_to_structure(
        seq, 1, dataset="hdfs", hdfs_feature_contract="notebook_raw_v1"
    )
    assert list(structure["cluster_ids"]) == [1, 2]
    assert structure["node_extra"].shape == (2, NODE_EXTRA_DIM)
    assert structure["edge_attr"].shape[1] == STRUCTURE_EDGE_DIM
    assert structure["y"] == 1
    assert structure["id_col"] == "block_id"


def test_shared_structure_splices_different_embeddings() -> None:
    sequences = {"blk_a": _hdfs_frame("blk_a", [7, 8])}
    labels = {"blk_a": 0}
    structures, stats = build_graph_structures(
        sequences,
        labels,
        dataset="hdfs",
        on_graph_error="fail",
        hdfs_feature_contract="notebook_raw_v1",
    )
    assert stats["n_graphs"] == 1
    extra = structures[0]["node_extra"].copy()
    tfidf = {7: np.ones(4, dtype=np.float32), 8: np.full(4, 2.0, dtype=np.float32)}
    sbert = {7: np.full(3, 9.0, dtype=np.float32), 8: np.full(3, 8.0, dtype=np.float32)}
    x_tfidf = _node_features_from_structure(structures[0], tfidf, 4, "fail")
    x_sbert = _node_features_from_structure(structures[0], sbert, 3, "fail")
    np.testing.assert_array_equal(x_tfidf[:, 4:], extra)
    np.testing.assert_array_equal(x_sbert[:, 3:], extra)
    np.testing.assert_array_equal(x_tfidf[:, :4], [[1, 1, 1, 1], [2, 2, 2, 2]])
    np.testing.assert_array_equal(x_sbert[:, :3], [[9, 9, 9], [8, 8, 8]])


def test_no_edge_features_is_first_column_slice() -> None:
    structure = _seq_to_structure(
        _hdfs_frame("blk_a", [1, 2, 3]),
        0,
        dataset="hdfs",
        hdfs_feature_contract="notebook_raw_v1",
    )
    sliced = structure["edge_attr"][:, :1]
    assert sliced.shape[1] == 1
    np.testing.assert_allclose(sliced.reshape(-1), structure["edge_attr"][:, 0])


def test_feature_group_ablation_removes_position_and_time_columns() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.modules.dataset import attach_cluster_embeddings

    structure = _seq_to_structure(
        _hdfs_frame("blk_a", [1, 2, 1]),
        0,
        dataset="hdfs",
        hdfs_feature_contract="notebook_raw_v1",
    )
    embeddings = {1: np.ones(4, dtype=np.float32), 2: np.full(4, 2.0, dtype=np.float32)}
    graph = attach_cluster_embeddings(
        [structure],
        embeddings,
        include_node_positional_features=False,
        include_edge_temporal_features=False,
        include_edge_positional_features=False,
        missing_embedding="fail",
    )[0]
    assert graph.x.shape[1] == 4 + 4
    assert graph.edge_attr.shape[1] == 1


def test_feature_contract_changes_node_extras() -> None:
    seq = _hdfs_frame("blk_a", [1, 1, 1])
    raw = _seq_to_structure(seq, 0, dataset="hdfs", hdfs_feature_contract="notebook_raw_v1")
    log = _seq_to_structure(seq, 0, dataset="hdfs", hdfs_feature_contract="stabilized_v2")
    assert raw["node_extra"][0, 0] == 3.0
    assert log["node_extra"][0, 0] == np.float32(np.log1p(3.0))


def test_graph_structure_roundtrip(tmp_path) -> None:
    sequences = {"blk_a": _hdfs_frame("blk_a", [1, 2])}
    structures, _ = build_graph_structures(
        sequences,
        {"blk_a": 1},
        dataset="hdfs",
        on_graph_error="fail",
        hdfs_feature_contract="notebook_raw_v1",
    )
    path = tmp_path / "graph_structure.pkl"
    save_graph_structures(path, structures, {"n_raw": 1, "n_unique": 1})
    loaded, meta = load_graph_structures(path)
    assert meta["n_unique"] == 1
    np.testing.assert_array_equal(loaded[0]["cluster_ids"], structures[0]["cluster_ids"])
    np.testing.assert_array_equal(loaded[0]["node_extra"], structures[0]["node_extra"])


def test_save_graph_dataset_with_zero_edge_graphs(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.modules.dataset import attach_cluster_embeddings, save_graph_dataset

    with_edge = _seq_to_structure(
        _hdfs_frame("blk_a", [1, 2]),
        0,
        dataset="hdfs",
        hdfs_feature_contract="notebook_raw_v1",
    )
    isolated = _seq_to_structure(
        _hdfs_frame("blk_b", [3]),
        1,
        dataset="hdfs",
        hdfs_feature_contract="notebook_raw_v1",
    )
    assert isolated["edge_index"].shape[1] == 0
    embeddings = {
        1: np.ones(4, dtype=np.float32),
        2: np.full(4, 2.0, dtype=np.float32),
        3: np.full(4, 3.0, dtype=np.float32),
    }
    graphs = attach_cluster_embeddings(
        [with_edge, isolated],
        embeddings,
        missing_embedding="fail",
    )
    path = tmp_path / "graph_dataset.pt"
    save_graph_dataset(
        graphs,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([], dtype=np.int64),
        path,
        node_dim=int(graphs[0].x.shape[1]),
        edge_dim=int(graphs[0].edge_attr.shape[1]),
        embed_dim=4,
    )
    assert path.exists()
