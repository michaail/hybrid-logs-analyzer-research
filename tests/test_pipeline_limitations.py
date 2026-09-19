"""Tests for the nine notebook-era pipeline limitations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modules.dataset import (
    MissingClusterEmbedding,
    MissingSequenceLabel,
    _node_features_from_structure,
    _seq_to_structure,
    build_graph_structures,
    compute_embeddings,
    densify_tfidf_rows,
)
from src.modules.graph_builder import select_unique_sequences
from src.modules.parser.drain_parser import OOV_CLUSTER_ID
from src.modules.sequencer import (
    BGL_SPLIT_STRIDE,
    bgl_split_name,
    build_sequences,
    event_count_slices,
    split_indices_from_bgl_windows,
)
from src.modules.unit_split import (
    bgl_time_split_boundaries,
    bgl_train_time_filter,
    indices_from_unit_split,
    stratified_id_split,
)


def _hdfs_frame(block_id: str, cluster_ids: list[int]) -> pd.DataFrame:
    n = len(cluster_ids)
    return pd.DataFrame(
        {
            "block_id": [block_id] * n,
            "cluster_id": cluster_ids,
            "parameters": [[] for _ in cluster_ids],
            "timestamp": [pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=i) for i in range(n)],
        }
    )


def test_missing_hdfs_label_is_not_normal() -> None:
    sequences = {"blk_a": _hdfs_frame("blk_a", [1, 2])}
    with pytest.raises(MissingSequenceLabel, match="blk_a"):
        build_graph_structures(sequences, {}, dataset="hdfs", on_graph_error="fail")


def test_select_unique_sequences_requires_every_label() -> None:
    sequences = {
        "blk_a": _hdfs_frame("blk_a", [1, 2]),
        "blk_b": _hdfs_frame("blk_b", [2, 1]),
    }
    with pytest.raises(KeyError, match="no label"):
        select_unique_sequences(sequences, {"blk_a": 0})


def test_select_unique_sequences_prefers_train_ids() -> None:
    sequences = {
        "blk_z": _hdfs_frame("blk_z", [1, 2, 3]),
        "blk_a": _hdfs_frame("blk_a", [1, 2, 3]),
    }
    labels = {"blk_z": 0, "blk_a": 0}
    unique, _, _ = select_unique_sequences(sequences, labels, prefer_ids={"blk_z"})
    assert list(unique) == ["blk_z"]


def test_run_ablation_sequence_labels_fail_closed(tmp_path: Path) -> None:
    import run_ablation

    sequences = {"blk_missing": _hdfs_frame("blk_missing", [1])}
    labels_path = tmp_path / "anomaly_label.csv"
    labels_path.write_text("BlockId,Label\nblk_other,Normal\n")
    with pytest.raises(KeyError, match="blk_missing"):
        run_ablation._sequence_labels(
            dataset="hdfs",
            sequences=sequences,
            labels_path=labels_path,
        )


def test_missing_embedding_fail_raises() -> None:
    structure = _seq_to_structure(
        _hdfs_frame("blk_a", [1, 2]),
        0,
        dataset="hdfs",
        hdfs_feature_contract="notebook_raw_v1",
    )
    with pytest.raises(MissingClusterEmbedding, match="2"):
        _node_features_from_structure(structure, {1: np.ones(3, dtype=np.float32)}, 3, "fail")


def test_tfidf_train_only_does_not_fit_held_out_tokens() -> None:
    templates = [
        {"cluster_id": 0, "template": "INFO receiving block BLK"},
        {"cluster_id": 1, "template": "ERROR unique_heldout_token BLK"},
    ]
    embeddings, cids, vectorizer = compute_embeddings(
        templates,
        {},
        tfidf_enabled=True,
        sbert_enabled=False,
        tfidf_fit_texts=["INFO receiving block BLK"],
    )
    assert cids == [0, 1]
    vocab = vectorizer.vocabulary_
    assert "unique_heldout_token" not in vocab
    assert embeddings.shape[0] == 2
    assert embeddings[1].sum() >= 0.0


def test_densify_tfidf_keeps_template_table_float32() -> None:
    from scipy.sparse import csr_matrix

    dense = densify_tfidf_rows(csr_matrix([[1.0, 0.0], [0.0, 2.0]]))
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense, [[1.0, 0.0], [0.0, 2.0]])


def test_bgl_time_split_windows_do_not_cross_the_cut() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "unix_ts": np.arange(1_000, 1_000 + n * 60, 60, dtype=np.int64),
            "cluster_id": [1] * n,
            "parameters": [[] for _ in range(n)],
            "is_anomaly": [False] * n,
        }
    )
    sequences = build_sequences(
        frame,
        "bgl",
        window_minutes=5,
        step_minutes=5,
        split="time",
        train_ratio=0.5,
        val_ratio=0.25,
    )
    assert sequences
    names = {bgl_split_name(wid) for wid in sequences}
    assert names == {"train", "val", "test"}
    n_train, n_val, _ = event_count_slices(n, 0.5, 0.25)
    train_end = int(frame["unix_ts"].iloc[n_train - 1])
    val_end = int(frame["unix_ts"].iloc[n_train + n_val - 1])
    for wid, group in sequences.items():
        name = bgl_split_name(int(wid))
        ts = group["unix_ts"].astype(int)
        if name == "train":
            assert int(ts.max()) <= train_end
        elif name == "val":
            assert int(ts.min()) > train_end
            assert int(ts.max()) <= val_end
        else:
            assert int(ts.min()) > val_end


def test_bgl_time_split_embargo_removes_boundary_events() -> None:
    n = 80
    frame = pd.DataFrame(
        {
            "unix_ts": np.arange(1_000, 1_000 + n * 60, 60, dtype=np.int64),
            "cluster_id": [1] * n,
            "parameters": [[] for _ in range(n)],
            "is_anomaly": [False] * n,
        }
    )
    sequences = build_sequences(
        frame,
        "bgl",
        window_minutes=5,
        step_minutes=5,
        split="time",
        train_ratio=0.5,
        val_ratio=0.25,
        embargo_minutes=2,
    )
    train_cut = int(frame["unix_ts"].iloc[39])
    val_cut = int(frame["unix_ts"].iloc[59])
    for window_id, group in sequences.items():
        timestamps = group["unix_ts"].astype(int)
        if bgl_split_name(int(window_id)) == "train":
            assert int(timestamps.max()) <= train_cut - 120
        elif bgl_split_name(int(window_id)) == "val":
            assert int(timestamps.min()) > train_cut + 120
            assert int(timestamps.max()) <= val_cut - 120
        else:
            assert int(timestamps.min()) > val_cut + 120


def test_bgl_parser_fit_boundary_uses_time_not_file_order(tmp_path: Path) -> None:
    rows = [
        f"- {timestamp} 2005.06.03 node 2005-06-03-00.00.00.000000 node RAS KERNEL INFO message\n"
        for timestamp in (50, 10, 20, 30, 40, 60, 70, 80, 90, 100)
    ]
    log_path = tmp_path / "bgl.log"
    log_path.write_text("".join(rows))
    train_cut, _ = bgl_time_split_boundaries(log_path, train_ratio=0.5, val_ratio=0.2)
    include = bgl_train_time_filter(train_cut - 10)
    kept = [int(row.split()[1]) for row in rows if include(row)]
    assert train_cut == 50
    assert kept == [10, 20, 30, 40]


def test_bgl_window_ids_namespace_splits() -> None:
    ids = [10, BGL_SPLIT_STRIDE + 10, 2 * BGL_SPLIT_STRIDE + 10]
    train, val, test = split_indices_from_bgl_windows(ids)
    np.testing.assert_array_equal(train, [0])
    np.testing.assert_array_equal(val, [1])
    np.testing.assert_array_equal(test, [2])


def test_unit_split_maps_block_ids() -> None:
    ids = [f"blk_{i}" for i in range(20)]
    labels = {item: int(i >= 10) for i, item in enumerate(ids)}
    split = stratified_id_split(ids, labels, seed=42)
    assert set(split["train"] + split["val"] + split["test"]) == set(ids)
    train, val, test = indices_from_unit_split(
        [str(item) for item in split["train"] + split["val"] + split["test"]],
        split,
    )
    assert len(train) + len(val) + len(test) == 20
    assert len(train) and len(val) and len(test)


def test_oov_cluster_id_is_negative() -> None:
    assert OOV_CLUSTER_ID < 0


def test_unmatched_oov_assigns_negative_cluster(tmp_path: Path) -> None:
    from src.modules.parser.drain_parser import DrainParser

    log_path = tmp_path / "tiny.log"
    log_path.write_text("081109 203615 148 INFO dfs.DataNode: Receiving block blk_1\n")
    parser = DrainParser()
    frame = parser.annotate_file(str(log_path), unmatched="oov")
    assert int(frame["cluster_id"].iloc[0]) == OOV_CLUSTER_ID


@pytest.mark.ml
def test_structure_decoder_is_directed() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.modules.models.gae import AttributeAwareGAE

    model = AttributeAwareGAE(node_dim=4, edge_dim=2, hidden_dim=8, latent_dim=4, structure_decoder="mlp")
    z = torch.randn(3, 4)
    forward = torch.tensor([[0, 1], [1, 0]])
    reverse = torch.tensor([[1, 0], [0, 1]])
    a = model.decode_structure(z, forward)
    b = model.decode_structure(z, reverse)
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-6)

    symmetric = AttributeAwareGAE(
        node_dim=4, edge_dim=2, hidden_dim=8, latent_dim=4, structure_decoder="inner_product"
    )
    assert torch.allclose(symmetric.decode_structure(z, forward), symmetric.decode_structure(z, reverse))


@pytest.mark.ml
def test_per_graph_negatives_stay_inside_each_graph() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.modules.models.gae import per_graph_negative_sampling

    # Two 4-node graphs with one edge each: 0→1 and 4→5.
    edge_index = torch.tensor([[0, 4], [1, 5]])
    ptr = torch.tensor([0, 4, 8])
    neg = per_graph_negative_sampling(edge_index, ptr)
    assert neg.size(0) == 2
    assert neg.size(1) > 0
    for src, dst in neg.t().tolist():
        assert (src < 4 and dst < 4) or (src >= 4 and dst >= 4)


@pytest.mark.ml
def test_eval_structure_score_includes_non_edges() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from src.modules.models.gae import AttributeAwareGAE, compute_anomaly_scores

    graph = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn(2, 2),
        y=torch.tensor([0]),
        num_nodes=3,
    )
    loader = DataLoader([graph, graph], batch_size=2)
    model = AttributeAwareGAE(node_dim=5, edge_dim=2, hidden_dim=8, latent_dim=4)
    model.eval()
    with_neg, _ = compute_anomaly_scores(
        model, loader, torch.device("cpu"), include_structure_non_edges=True, alpha=1, beta=0, gamma=0
    )
    pos_only, _ = compute_anomaly_scores(
        model, loader, torch.device("cpu"), include_structure_non_edges=False, alpha=1, beta=0, gamma=0
    )
    assert with_neg.shape == pos_only.shape == (2,)
    assert not np.allclose(with_neg, pos_only)


@pytest.mark.ml
def test_one_epoch_smoke_on_tiny_batch() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from torch.optim import Adam
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from src.modules.models.gae import AttributeAwareGAE, compute_anomaly_scores, train_epoch

    graphs = [
        Data(
            x=torch.randn(4, 6),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.randn(3, 3),
            y=torch.tensor([label]),
            num_nodes=4,
        )
        for label in (0, 0, 1)
    ]
    loader = DataLoader(graphs, batch_size=2, shuffle=True)
    model = AttributeAwareGAE(node_dim=6, edge_dim=3, hidden_dim=8, latent_dim=4)
    opt = Adam(model.parameters(), lr=0.01)
    total, structure, node, edge = train_epoch(model, loader, opt, torch.device("cpu"))
    assert np.isfinite(total)
    assert np.isfinite(structure) and structure >= 0
    scores, labels = compute_anomaly_scores(model, loader, torch.device("cpu"))
    assert scores.shape == labels.shape == (3,)


@pytest.mark.ml
def test_without_sbert_decoder_skips_sbert_columns() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from torch.optim import Adam
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from src.modules.models.gae import (
        AttributeAwareGAE,
        compute_anomaly_scores,
        node_recon_index_tensor,
        train_epoch,
    )

    node_dim = 16
    sbert_dim = 4
    index = node_recon_index_tensor(node_dim, sbert_dim=sbert_dim)
    assert index.numel() == node_dim - sbert_dim
    graphs = [
        Data(
            x=torch.randn(4, node_dim),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.randn(3, 3),
            y=torch.tensor([0]),
            num_nodes=4,
        )
    ]
    loader = DataLoader(graphs, batch_size=1)
    model = AttributeAwareGAE(
        node_dim=node_dim,
        edge_dim=3,
        hidden_dim=8,
        latent_dim=4,
        node_recon_index=index,
    )
    assert model.node_decoder[2].out_features == node_dim - sbert_dim
    opt = Adam(model.parameters(), lr=0.01)
    total, structure, node, edge = train_epoch(model, loader, opt, torch.device("cpu"))
    assert np.isfinite(total) and np.isfinite(node)
    scores, labels = compute_anomaly_scores(model, loader, torch.device("cpu"))
    assert scores.shape == labels.shape == (1,)


@pytest.mark.ml
def test_projected_gated_fusion_and_balanced_node_loss() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from torch.optim import Adam
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from src.modules.models.gae import AttributeAwareGAE, compute_anomaly_scores, train_epoch

    graph = Data(
        x=torch.randn(5, 12),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.randn(4, 2),
        y=torch.tensor([0]),
        num_nodes=5,
    )
    loader = DataLoader([graph], batch_size=1)
    model = AttributeAwareGAE(
        node_dim=12,
        edge_dim=2,
        hidden_dim=8,
        latent_dim=4,
        tfidf_dim=4,
        sbert_dim=5,
        fusion_mode="projected_gated",
        modality_projection_dim=6,
        node_loss_mode="block_balanced",
    )
    assert set(model.modality_projectors) == {"tfidf", "sbert", "extras"}
    optimizer = Adam(model.parameters(), lr=0.01)
    total, _, node, _ = train_epoch(model, loader, optimizer, torch.device("cpu"))
    assert np.isfinite(total) and np.isfinite(node)
    _, _, components, _ = compute_anomaly_scores(
        model, loader, torch.device("cpu"), return_components=True
    )
    assert components["tfidf"].shape == (1,)
    assert components["sbert"].shape == (1,)
    assert components["extras"].shape == (1,)


@pytest.mark.ml
def test_graph_directory_dataset_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    from src.modules.dataset import GraphDirectoryDataset, save_graph_directory

    graphs = [torch.tensor([index], dtype=torch.long) for index in (1, 2)]
    directory = save_graph_directory(graphs, tmp_path / "graphs")
    loaded = GraphDirectoryDataset(directory)
    assert len(loaded) == 2
    assert int(loaded[0].item()) == 1
    assert int(loaded[1].item()) == 2
