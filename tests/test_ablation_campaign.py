from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.modules.ablation import (
    GraphIdentityError,
    LEGACY_GRAPH_IDENTITY,
    apply_split_lock,
    assert_train_only_compatible,
    component_metrics,
    enabled_experiments,
    experiment_requires_graph_rebuild,
    feature_contract_from_config,
    graph_identity_from_config,
    graph_identity_from_meta,
    gzip_file,
    gunzip_file,
    identities_match,
    load_matrix,
    load_split_lock,
    save_split_lock,
    unique_sequences_from_config,
    write_campaign_report,
    write_eval_pack,
)
from src.modules.artifacts import fingerprint
from src.modules.graph_builder import select_unique_sequences

REPOSITORY_ROOT = Path(__file__).parents[1]


def _baseline_config(**overrides: object) -> dict:
    config = yaml.safe_load((REPOSITORY_ROOT / "configs" / "ablation_base.yaml").read_text())
    config["experiment"]["dataset"] = "hdfs"
    ablation = config["ablation"]
    for key, value in overrides.items():
        if key == "tfidf_enabled":
            ablation["embeddings"]["tfidf_enabled"] = value
        elif key == "sbert_enabled":
            ablation["embeddings"]["sbert_enabled"] = value
        elif key == "use_edge_features":
            ablation["graph"]["use_edge_features"] = value
        elif key == "unique_sequences":
            ablation["graph"]["unique_sequences"] = value
        else:
            ablation[key] = value
    return config


def test_family_yaml_files_are_split() -> None:
    representation = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_representation.yaml")
    train = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_train.yaml")
    mixed = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_matrix.yaml")
    assert representation["family"] == "representation"
    assert representation["requires_graph_rebuild"] is True
    names_a = {item["name"] for item in enabled_experiments(representation)}
    assert {"baseline_full", "tfidf_only", "no_llm_enrichment", "no_enrichment"} <= names_a
    assert all(experiment_requires_graph_rebuild(item, representation) for item in enabled_experiments(representation))
    names_b = {item["name"] for item in enabled_experiments(train)}
    assert {"alpha_0", "gine_mean_agg", "latent_32", "inner_product_structure"} <= names_b
    assert not any(experiment_requires_graph_rebuild(item, train) for item in enabled_experiments(train))
    mixed_names = {item["name"] for item in enabled_experiments(mixed)}
    assert "tfidf_only" not in mixed_names
    assert mixed["requires_graph_rebuild"] is False


def test_bgl_representation_yaml_omits_hdfs_feature_contract() -> None:
    bgl = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_representation_bgl.yaml")
    assert bgl["family"] == "representation"
    names = {item["name"] for item in enabled_experiments(bgl)}
    assert "feature_contract_stabilized_v2" not in names
    assert {"baseline_full", "tfidf_only", "no_llm_enrichment", "sbert_only", "no_edge_features"} <= names
    assert "no_enrichment" not in names
    assert all(experiment_requires_graph_rebuild(item, bgl) for item in enabled_experiments(bgl))
    base = yaml.safe_load((REPOSITORY_ROOT / "configs" / "ablation_base.yaml").read_text())
    assert base["parser"]["hdfs"]["raw_file"] == "hdfs/HDFS_full.log"
    assert base["parser"]["bgl"]["raw_file"] == "bgl/BGL_full.log"
    assert base["sequencing"]["bgl"]["step_minutes"] == base["sequencing"]["bgl"]["window_minutes"]
    assert base["sequencing"]["bgl"]["split"] == "time"
    assert base["ablation"]["representation"]["fit_on"] == "train_only"
    assert base["ablation"]["graph"]["structure_decoder"] == "mlp"
    prepare_hdfs = (REPOSITORY_ROOT / "scripts" / "prepare_hdfs_campaign.py").read_text()
    assert "parser.hdfs.raw_file=hdfs/HDFS_full.log" in prepare_hdfs
    assert "ablation.enrichment_model_size=large" in prepare_hdfs
    assert "ablation.graph.unique_sequences=true" in prepare_hdfs
    prepare = (REPOSITORY_ROOT / "scripts" / "prepare_bgl_campaign.py").read_text()
    assert "experiment.dataset=bgl" in prepare
    assert "ablation_representation_bgl.yaml" in prepare
    assert "ablation.enrichment_model_size=large" in prepare
    run_ablation_src = (REPOSITORY_ROOT / "run_ablation.py").read_text()
    assert 'else "both"' not in run_ablation_src
    assert "deepseek-v4-pro" in run_ablation_src
    enrichment_src = (REPOSITORY_ROOT / "src" / "modules" / "enrichment.py").read_text()
    assert "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO" in enrichment_src
    assert "AZURE_OPENAI_DEPLOYMENT_MISTRAL_LARGE" not in enrichment_src
    assert "AZURE_OPENAI_DEPLOYMENT_MISTRAL_SMALL" not in enrichment_src
    llm_src = (REPOSITORY_ROOT / "src" / "modules" / "enricher" / "llm.py").read_text()
    assert "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO" in llm_src


def test_enrichment_rejects_small_and_both_models() -> None:
    from src.modules.enrichment import enrich_templates

    with pytest.raises(ValueError, match="Deepseek"):
        enrich_templates([{"template": "x", "cluster_id": 0}], "hdfs", model_size="small")
    with pytest.raises(ValueError, match="Deepseek"):
        enrich_templates([{"template": "x", "cluster_id": 0}], "hdfs", model_size="both")


def test_optional_workspace_relative_allows_resume_without_templates() -> None:
    import run_ablation

    workspace = REPOSITORY_ROOT
    assert run_ablation._optional_workspace_relative(None, workspace, "cached/templates.json") == (
        "cached/templates.json"
    )
    assert run_ablation._optional_workspace_relative(None, workspace) is None
    relative = run_ablation._optional_workspace_relative(
        workspace / "configs" / "ablation_base.yaml", workspace
    )
    assert relative == "configs/ablation_base.yaml"


def test_feature_contract_defaults_to_notebook_raw() -> None:
    config = _baseline_config()
    assert feature_contract_from_config(config) == "notebook_raw_v1"
    identity = graph_identity_from_config(config)
    assert identity["unique_sequences"] is True
    assert identity["fit_on"] == "train_only"
    assert identity["split_protocol"] == "stratified"
    assert identities_match(
        identity,
        {
            **LEGACY_GRAPH_IDENTITY,
            "unique_sequences": True,
            "fit_on": "train_only",
            "split_protocol": "stratified",
        },
    )


def test_train_only_guard_allows_legacy_baseline_without_meta() -> None:
    config = _baseline_config(unique_sequences=False)
    config["ablation"]["representation"] = {"fit_on": "all"}
    assert_train_only_compatible(config, bundle_meta=None)


def test_train_only_guard_rejects_representation_change_on_legacy_graph() -> None:
    config = _baseline_config(tfidf_enabled=True, sbert_enabled=False)
    with pytest.raises(GraphIdentityError, match="Family A"):
        assert_train_only_compatible(config, bundle_meta=None)


def test_train_only_guard_rejects_identity_mismatch() -> None:
    config = _baseline_config()
    meta = {
        "llm_enrichment_enabled": False,
        "tfidf_enabled": True,
        "sbert_enabled": True,
        "use_edge_features": True,
        "feature_contract": "notebook_raw_v1",
    }
    with pytest.raises(GraphIdentityError, match="mismatch"):
        assert_train_only_compatible(config, bundle_meta=meta)


def test_split_lock_roundtrip(tmp_path: Path) -> None:
    idx_train = np.array([0, 1, 2], dtype=np.int64)
    idx_val = np.array([3], dtype=np.int64)
    idx_test = np.array([4], dtype=np.int64)
    path = tmp_path / "split_lock.npz"
    lock_id = save_split_lock(
        path,
        idx_train,
        idx_val,
        idx_test,
        sequence_ids=["a", "b", "c", "d", "e"],
        seed=42,
    )
    loaded = load_split_lock(path)
    assert loaded["split_lock_id"] == lock_id
    assert loaded["n_total"] == 5
    class _Graph:
        def __init__(self, name: str) -> None:
            self.block_id = name

    graphs = [_Graph(name) for name in ("a", "b", "c", "d", "e")]
    train, val, test = apply_split_lock(graphs, loaded, sequence_ids=["a", "b", "c", "d", "e"])
    np.testing.assert_array_equal(train, idx_train)
    np.testing.assert_array_equal(val, idx_val)
    np.testing.assert_array_equal(test, idx_test)


def test_split_lock_rejects_length_mismatch() -> None:
    lock = {
        "idx_train": np.array([0]),
        "idx_val": np.array([1]),
        "idx_test": np.array([2]),
        "n_total": 3,
        "split_lock_id": "x",
    }
    with pytest.raises(ValueError, match="n_total"):
        apply_split_lock([object(), object()], lock)


def test_gzip_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "graph_dataset.pt"
    source.write_bytes(b"graph-bytes" * 100)
    compressed = gzip_file(source, tmp_path / "graph_dataset.pt.gz")
    restored = gunzip_file(compressed, tmp_path / "restored.pt")
    assert restored.read_bytes() == source.read_bytes()


def test_eval_pack_and_campaign_report(tmp_path: Path) -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    structure = np.array([0.4, 0.4, 0.4, 0.4])
    node = np.array([0.1, 0.15, 0.7, 0.8])
    edge = np.array([0.05, 0.1, 0.5, 0.6])
    run_dir = tmp_path / "outputs" / "hdfs" / "camp_baseline_full"
    written = write_eval_pack(
        run_dir,
        metrics={"test_f1": 0.9, "test_pr_auc": 0.91, "test_roc_auc": 0.95, "val_f1": 0.88},
        history_epochs=[
            {
                "epoch": 1,
                "train_total_loss": 1.0,
                "train_structure_loss": 0.5,
                "train_node_loss": 0.3,
                "train_edge_loss": 0.2,
                "val_f1": 0.5,
                "val_pr_auc": 0.4,
                "val_roc_auc": 0.6,
            }
        ],
        labels=labels,
        scores=scores,
        structure=structure,
        node=node,
        edge=edge,
        threshold=0.5,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        campaign_meta={"campaign_id": "camp", "experiment_name": "baseline_full", "family": "train"},
    )
    assert written["component_metrics"].exists()
    assert (run_dir / "figures" / "training_history.png").exists()
    assert (run_dir / "scores" / "test_component_scores.csv").exists()
    components = component_metrics(
        labels,
        combined=scores,
        structure=structure,
        node=node,
        edge=edge,
        predictions=(scores > 0.5).astype(int),
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )
    assert components["n_true_positives"] == 2
    summary = write_campaign_report(tmp_path, dataset="hdfs", campaign_id="camp")
    assert summary.exists()
    leaderboard = json.loads(summary.read_text())
    assert leaderboard[0]["name"] == "baseline_full"
    assert (tmp_path / "outputs" / "hdfs" / "campaigns" / "camp" / "README.md").exists()
    assert (tmp_path / "outputs" / "hdfs" / "campaigns" / "camp" / "ablation_comparison.png").exists()


def test_fingerprint_ignores_git_revision() -> None:
    inputs = [{"path": "a", "size_bytes": 1, "modified_ns": 2}]
    first = fingerprint(config={"x": 1}, inputs=inputs, revision="aaa")
    second = fingerprint(config={"x": 1}, inputs=inputs, revision="bbb")
    third = fingerprint(config={"x": 2}, inputs=inputs, revision="aaa")
    assert first == second
    assert first != third


def _sequence_frame(block_id: str, cluster_ids: list[int]):
    import pandas as pd

    return pd.DataFrame({"block_id": [block_id] * len(cluster_ids), "cluster_id": cluster_ids})


def test_select_unique_sequences_collapses_identical_order() -> None:
    sequences = {
        "blk_z": _sequence_frame("blk_z", [1, 2, 3]),
        "blk_a": _sequence_frame("blk_a", [1, 2, 3]),
        "blk_b": _sequence_frame("blk_b", [3, 2, 1]),
    }
    labels = {"blk_z": 0, "blk_a": 0, "blk_b": 0}
    unique, unique_labels, stats = select_unique_sequences(sequences, labels)
    assert list(unique) == ["blk_a", "blk_b"]
    assert unique_labels == {"blk_a": 0, "blk_b": 0}
    assert stats == {"n_raw": 3, "n_unique": 2, "n_mixed_label_fingerprints": 0}


def test_select_unique_sequences_keeps_both_labels_when_mixed() -> None:
    sequences = {
        "blk_c": _sequence_frame("blk_c", [1, 1]),
        "blk_a": _sequence_frame("blk_a", [1, 1]),
        "blk_b": _sequence_frame("blk_b", [1, 1]),
    }
    labels = {"blk_a": 1, "blk_b": 0, "blk_c": 0}
    unique, unique_labels, stats = select_unique_sequences(sequences, labels)
    assert list(unique) == ["blk_a", "blk_b"]
    assert unique_labels == {"blk_a": 1, "blk_b": 0}
    assert stats["n_raw"] == 3
    assert stats["n_unique"] == 2
    assert stats["n_mixed_label_fingerprints"] == 1


def test_hdfs_unique_sequences_identity_and_legacy_default() -> None:
    config = _baseline_config()
    assert unique_sequences_from_config(config) is True
    assert graph_identity_from_config(config)["unique_sequences"] is True
    assert graph_identity_from_config(config)["fit_on"] == "train_only"
    missing = graph_identity_from_meta(
        {
            "llm_enrichment_enabled": True,
            "enrichment_model_size": "large",
            "tfidf_enabled": True,
            "sbert_enabled": True,
            "use_edge_features": True,
            "feature_contract": "notebook_raw_v1",
        }
    )
    assert missing is not None
    assert missing["unique_sequences"] is False
    assert identities_match(missing, LEGACY_GRAPH_IDENTITY)
    with pytest.raises(GraphIdentityError, match="mismatch"):
        assert_train_only_compatible(config, bundle_meta=missing)
    bgl = yaml.safe_load((REPOSITORY_ROOT / "configs" / "ablation_base.yaml").read_text())
    assert unique_sequences_from_config(bgl) is False
    assert graph_identity_from_config(bgl)["unique_sequences"] is False
    assert graph_identity_from_config(bgl)["split_protocol"] == "time"
    baseline = yaml.safe_load((REPOSITORY_ROOT / "configs" / "hdfs_baseline.yaml").read_text())
    assert unique_sequences_from_config(baseline) is False


def test_stage45_config_includes_unique_sequences_flag() -> None:
    import run_ablation

    enabled = run_ablation._stage45_config(_baseline_config())
    disabled = run_ablation._stage45_config(_baseline_config(unique_sequences=False))
    assert enabled["unique_sequences"] is True
    assert disabled["unique_sequences"] is False
    assert enabled != disabled


def test_stage45_structure_config_ignores_embeddings_and_edge_flag() -> None:
    import run_ablation

    hybrid = run_ablation._stage45_structure_config(_baseline_config())
    tfidf = run_ablation._stage45_structure_config(
        _baseline_config(tfidf_enabled=True, sbert_enabled=False)
    )
    no_llm = run_ablation._stage45_structure_config(_baseline_config(llm_enrichment_enabled=False))
    no_edges = run_ablation._stage45_structure_config(_baseline_config(use_edge_features=False))
    assert hybrid == tfidf == no_llm == no_edges
    assert "embeddings" not in hybrid
    assert "use_edge_features" not in hybrid
    assert "llm_enrichment_enabled" not in hybrid
    assert hybrid["fit_on"] == "train_only"
    unique_off = run_ablation._stage45_structure_config(_baseline_config(unique_sequences=False))
    assert unique_off != hybrid
    stabilized = _baseline_config()
    stabilized["ablation"]["feature_contract"] = "stabilized_v2"
    assert run_ablation._stage45_structure_config(stabilized) != hybrid