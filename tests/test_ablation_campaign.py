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
    experiment_seed_pairs,
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
    _seeded_campaign_statistics,
)
from src.modules.artifacts import fingerprint
from src.modules.graph_builder import select_unique_sequences
from src.modules.unit_split import topology_grouped_id_split

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


def test_focused_hdfs_matrices_cover_proposed_experiment() -> None:
    representation = load_matrix(
        REPOSITORY_ROOT / "configs" / "ablation_hdfs_representation.yaml"
    )
    train = load_matrix(
        REPOSITORY_ROOT / "configs" / "ablation_hdfs_fusion_train.yaml"
    )
    assert len(representation["seeds"]) >= 5
    assert representation["seeds"] == train["seeds"]
    assert {item["name"] for item in enabled_experiments(representation)} == {
        "baseline_full", "tfidf_only", "sbert_only", "no_edge_features"
    }
    assert {item["name"] for item in enabled_experiments(train)} == {
        "hybrid_projected_gated", "hybrid_lexical_recon",
        "hybrid_block_balanced", "alpha_0",
    }
    pairs = experiment_seed_pairs(representation, 42)
    assert len(pairs) == 4 * len(representation["seeds"])


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
    assert base["ablation"]["representation"]["fit_on_by_dataset"]["bgl"] == "all"
    assert base["ablation"]["graph"]["structure_decoder"] == "mlp"
    prepare_hdfs = (REPOSITORY_ROOT / "scripts" / "prepare_hdfs_campaign.py").read_text()
    assert "parser.hdfs.raw_file=hdfs/HDFS_full.log" in prepare_hdfs
    assert "ablation.enrichment_model_size=large" in prepare_hdfs
    assert "ablation.graph.unique_sequences=true" not in prepare_hdfs
    prepare = (REPOSITORY_ROOT / "scripts" / "prepare_bgl_campaign.py").read_text()
    assert "experiment.dataset=bgl" in prepare
    assert "ablation_representation_bgl.yaml" in prepare
    assert "ablation.enrichment_model_size=large" in prepare
    run_ablation_src = (REPOSITORY_ROOT / "run_ablation.py").read_text()
    assert 'else "both"' not in run_ablation_src
    assert "deepseek-v4-pro" in run_ablation_src
    assert "enrichment_prompt_version" in run_ablation_src
    from src.modules.enrichment import ENRICHMENT_PROMPT_VERSION

    assert ENRICHMENT_PROMPT_VERSION == "bgl_extended_v1"
    enrichment_src = (REPOSITORY_ROOT / "src" / "modules" / "enrichment.py").read_text()
    assert "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO" in enrichment_src
    assert "AZURE_OPENAI_DEPLOYMENT_MISTRAL_LARGE" not in enrichment_src
    assert "AZURE_OPENAI_DEPLOYMENT_MISTRAL_SMALL" not in enrichment_src
    llm_src = (REPOSITORY_ROOT / "src" / "modules" / "enricher" / "llm.py").read_text()
    assert "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO" in llm_src


def test_bgl_closing_matrices_have_five_paired_seeds_and_controls() -> None:
    llm = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_bgl_llm_closed.yaml")
    pb3 = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_bgl_pb3.yaml")
    assert llm["seeds"] == pb3["seeds"] == [13, 29, 42, 71, 101]
    assert {item["name"] for item in enabled_experiments(llm)} == {
        "tfidf_only", "sbert_raw", "sbert_llm", "hybrid_raw", "hybrid_llm",
    }
    assert {item["name"] for item in enabled_experiments(pb3)} == {
        "no_positional_features", "no_temporal_features",
        "no_temporal_or_positional_features",
    }
    assert len(experiment_seed_pairs(llm, 42)) == 25
    assert len(experiment_seed_pairs(pb3, 42)) == 15


def test_graph_identity_tracks_feature_group_ablations() -> None:
    baseline = _baseline_config()
    no_position = _baseline_config()
    no_position["ablation"]["graph"]["node_positional_features"] = False
    no_position["ablation"]["graph"]["edge_positional_features"] = False
    assert not identities_match(
        graph_identity_from_config(baseline),
        graph_identity_from_config(no_position),
    )


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
    assert identity["unique_sequences"] is False
    assert identity["fit_on"] == "train_only"
    assert identity["split_protocol"] == "stratified"
    assert identities_match(
        identity,
        {
            **LEGACY_GRAPH_IDENTITY,
            "unique_sequences": False,
            "fit_on": "train_only",
            "split_protocol": "stratified",
        },
    )
    assert identity["sbert_text"] == "embedding_text"


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
        protocol="stratified",
    )
    loaded = load_split_lock(path)
    assert loaded["split_lock_id"] == lock_id
    assert loaded["n_total"] == 5
    assert loaded["protocol"] == "stratified"
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


def test_topology_grouped_split_has_no_group_overlap() -> None:
    ids = [f"b{index}" for index in range(120)]
    labels = {item: int(index % 11 == 0) for index, item in enumerate(ids)}
    fingerprints = {item: f"g{index // 3}" for index, item in enumerate(ids)}
    split = topology_grouped_id_split(ids, labels, fingerprints, seed=42)
    assert set().union(*map(set, split.values())) == set(ids)
    groups = {
        name: {fingerprints[item] for item in values}
        for name, values in split.items()
    }
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


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


def test_seeded_statistics_are_paired_against_baseline() -> None:
    import pandas as pd

    rows = []
    for seed, baseline, arm in ((1, 0.7, 0.8), (2, 0.8, 0.85), (3, 0.9, 0.95)):
        for name, value in (("baseline_full", baseline), ("arm", arm)):
            rows.append({
                "name": name, "seed": seed,
                "test_f1": value, "test_pr_auc": value, "test_roc_auc": value,
            })
    summary, paired = _seeded_campaign_statistics(
        pd.DataFrame(rows), "baseline_full", bootstrap_samples=200, bootstrap_seed=1
    )
    assert set(summary["name"]) == {"baseline_full", "arm"}
    arm_f1 = paired[(paired["name"] == "arm") & (paired["metric"] == "test_f1")].iloc[0]
    assert arm_f1["n_paired_seeds"] == 3
    assert arm_f1["mean_delta"] > 0


def test_seeded_statistics_require_same_frozen_split_lock() -> None:
    import pandas as pd

    frame = pd.DataFrame([
        {"name": "baseline_full", "seed": 1, "split_lock_id": "lock-a", "test_f1": 0.7, "test_pr_auc": 0.7, "test_roc_auc": 0.7},
        {"name": "arm", "seed": 1, "split_lock_id": "lock-b", "test_f1": 0.9, "test_pr_auc": 0.9, "test_roc_auc": 0.9},
    ])
    _, paired = _seeded_campaign_statistics(frame, "baseline_full", bootstrap_samples=20)
    assert paired[paired["name"] == "arm"].empty


def test_campaign_preparation_preserves_previously_prepared_graph_entries() -> None:
    source = (REPOSITORY_ROOT / "run_ablation.py").read_text()
    assert "merged_graphs = {**previous_graphs" in source
    assert "graphs=[merged_graphs[name] for name in sorted(merged_graphs)]" in source


def test_graph_structure_cache_schema_tracks_isolation_forest_event_ids() -> None:
    import run_ablation

    config = _baseline_config()
    assert run_ablation._stage45_structure_config(config)["structure_schema"] == "event_cluster_ids_v2"


def test_bgl_time_block_bootstrap_reports_paired_ap_uncertainty(tmp_path: Path) -> None:
    import pandas as pd

    from src.modules.ablation import _bgl_time_block_bootstrap

    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    graph_ids = [10, 86_410, 691_210, 777_610]
    for name, scores in (("baseline", [0.1, 0.2, 0.7, 0.8]), ("arm", [0.2, 0.3, 0.8, 0.9])):
        directory = tmp_path / name / "scores"
        directory.mkdir(parents=True)
        pd.DataFrame({"graph_id": graph_ids, "label": [0, 0, 1, 1], "score": scores}).to_csv(
            directory / "test_component_scores.csv", index=False
        )
    frame = _bgl_time_block_bootstrap([
        {"name": "hybrid_llm", "seed": 13, "run_dir": str(tmp_path / "baseline")},
        {"name": "arm", "seed": 13, "run_dir": str(tmp_path / "arm")},
    ], baseline_name="hybrid_llm", samples=40)
    assert set(frame["block_days"]) == {1, 7}
    assert (frame["bootstrap_unit"] == "paired_time_block").all()


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
    assert unique_sequences_from_config(config) is False
    assert graph_identity_from_config(config)["unique_sequences"] is False
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

    default = run_ablation._stage45_config(_baseline_config())
    topology_only = run_ablation._stage45_config(_baseline_config(unique_sequences=True))
    assert default["unique_sequences"] is False
    assert topology_only["unique_sequences"] is True
    assert default != topology_only


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
    topology_only = run_ablation._stage45_structure_config(_baseline_config(unique_sequences=True))
    assert topology_only != hybrid
    stabilized = _baseline_config()
    stabilized["ablation"]["feature_contract"] = "stabilized_v2"
    assert run_ablation._stage45_structure_config(stabilized) != hybrid


def test_missing_sbert_text_meta_matches_embedding_text() -> None:
    config = _baseline_config()
    meta = {
        "llm_enrichment_enabled": True,
        "enrichment_model_size": "large",
        "tfidf_enabled": True,
        "sbert_enabled": True,
        "use_edge_features": True,
        "feature_contract": "notebook_raw_v1",
        "unique_sequences": False,
        "fit_on": "train_only",
        "split_protocol": "stratified",
    }
    actual = graph_identity_from_meta(meta)
    assert actual is not None
    assert actual["sbert_text"] == "embedding_text"
    assert_train_only_compatible(config, bundle_meta=meta)
    grounded = dict(meta)
    grounded["sbert_text"] = "grounded_v1"
    with pytest.raises(GraphIdentityError, match="mismatch"):
        assert_train_only_compatible(config, bundle_meta=grounded)


def test_fusion_yaml_rebuilds_and_train_arm_does_not() -> None:
    fusion = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_fusion_bgl.yaml")
    train = load_matrix(REPOSITORY_ROOT / "configs" / "ablation_train.yaml")
    assert fusion["family"] == "representation"
    assert fusion["requires_graph_rebuild"] is True
    names = {item["name"] for item in enabled_experiments(fusion)}
    assert names == {"hybrid_grounded", "sbert_grounded"}
    assert all(experiment_requires_graph_rebuild(item, fusion) for item in enabled_experiments(fusion))
    hybrid = next(item for item in enabled_experiments(fusion) if item["name"] == "hybrid_grounded")
    sbert = next(item for item in enabled_experiments(fusion) if item["name"] == "sbert_grounded")
    assert hybrid["overrides"]["ablation.embeddings.sbert_text"] == "grounded_v1"
    assert sbert["overrides"]["ablation.embeddings.sbert_text"] == "grounded_v1"
    assert hybrid["overrides"]["ablation.fusion.node_reconstruct"] == "without_sbert"
    assert sbert["overrides"]["ablation.fusion.node_reconstruct"] == "all"
    train_names = {item["name"] for item in enabled_experiments(train)}
    assert "lexical_node_recon" in train_names
    lexical = next(item for item in enabled_experiments(train) if item["name"] == "lexical_node_recon")
    assert experiment_requires_graph_rebuild(lexical, train) is False
    assert lexical["overrides"]["ablation.fusion.node_reconstruct"] == "without_sbert"


def test_node_reconstruct_is_not_graph_identity() -> None:
    config = _baseline_config()
    config["ablation"]["fusion"] = {"node_reconstruct": "without_sbert"}
    identity = graph_identity_from_config(config)
    assert "node_reconstruct" not in identity
    assert identity["sbert_text"] == "embedding_text"


def test_bgl_uses_transductive_performance_fit_without_changing_hdfs_default() -> None:
    from src.modules.ablation import fit_on_from_config

    hdfs = _baseline_config()
    bgl = _baseline_config()
    bgl["experiment"]["dataset"] = "bgl"
    assert fit_on_from_config(hdfs) == "train_only"
    assert fit_on_from_config(bgl) == "all"