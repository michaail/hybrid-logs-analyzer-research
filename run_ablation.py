#!/usr/bin/env python3
"""Run reproducible HDFS/BGL ablations locally or from Google Colab.

The code checkout and the data workspace are intentionally independent:

* ``--code-root`` contains this repository and versioned configuration.
* ``--workspace-root`` contains ignored raw data, cache artifacts, models, and
  result files.  In Colab it normally points to ``/content/workspace``.

Stages publish only after all declared outputs exist, so a disconnected Colab
runtime can safely resume from the latest successful stage cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.modules.ablation import (
    GraphIdentityError,
    apply_split_lock,
    assert_train_only_compatible,
    enabled_experiments,
    experiment_seed_pairs,
    experiment_requires_graph_rebuild,
    feature_contract_from_config,
    file_digest,
    fit_on_from_config,
    fusion_mode_from_config,
    graph_identity_from_config,
    gunzip_file,
    gzip_file,
    load_bundle_meta,
    load_matrix,
    load_split_lock,
    node_reconstruct_from_config,
    node_loss_from_config,
    save_split_lock,
    sbert_text_from_config,
    sequence_ids_from_graphs,
    split_protocol_from_config,
    structure_decoder_from_config,
    unique_sequences_from_config,
    write_campaign_manifest,
    write_campaign_report,
    write_eval_pack,
)
from src.modules.artifacts import SUCCESS_FILE, ArtifactStore, fingerprint, git_revision, input_metadata
from src.modules.dataset import (
    _require_torch_geometric,
    attach_cluster_embeddings,
    build_graph_structures,
    compute_embeddings,
    embedding_block_dims,
    graph_split_dir,
    load_graph_splits,
    load_graph_structures,
    save_graph_dataset,
    save_graph_structures,
    sbert_dim_from_meta,
    split_dataset,
    split_label_indices,
)
from src.modules.enrichment import (
    ENRICHMENT_PROMPT_VERSION,
    enrichment_provenance,
    enrich_templates,
    load_enriched_templates,
    require_complete_enrichment,
    require_valid_enrichment_provenance,
)
from src.modules.graph_builder import select_unique_sequences
from src.modules.classical_baseline import window_feature_matrices
from src.modules.inference_release import (
    HdfsReleaseIdentity,
    HdfsReleaseSource,
    export_hdfs_inference_release,
)
from src.modules.parser import BGLParser, DrainParser, OOV_CLUSTER_ID, OOV_TEMPLATE
from src.modules.sequencer import (
    build_sequences,
    load_sequences,
    save_sequences,
    split_indices_from_bgl_windows,
)
from src.modules.unit_split import (
    bgl_time_split_boundaries,
    bgl_train_time_filter,
    hdfs_train_line_filter,
    hdfs_topology_fingerprints,
    indices_from_unit_split,
    save_unit_split,
    scan_hdfs_block_ids,
    stratified_id_split,
    topology_grouped_id_split,
)
from src.modules.utils import get_device, seed_everything

# Embedding-agnostic graph structures, reused across Family A arms in one process.
_GRAPH_STRUCTURE_MEMO: dict[str, tuple[list, dict[str, Any]]] = {}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and retain its source location."""
    config_path = Path(path).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["__pipeline__"] = {
        "config_path": str(config_path),
        "code_root": str(config_path.parent.parent),
        "workspace_root": os.environ.get("PIPELINE_WORKSPACE_ROOT"),
    }
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` YAML overrides without mutating *config*."""
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected KEY=VALUE override, received: {override!r}")
        dotted_key, raw_value = override.split("=", maxsplit=1)
        keys = dotted_key.split(".")
        if not all(keys):
            raise ValueError(f"Invalid override key: {dotted_key!r}")
        target: dict[str, Any] = updated
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            if not isinstance(target[key], dict):
                raise ValueError(f"Override path is not a mapping: {dotted_key!r}")
            target = target[key]
        target[keys[-1]] = yaml.safe_load(raw_value)
    return updated


def get_run_tag(config: dict[str, Any]) -> str:
    """Return an explicit run id or a timestamped, human-readable one."""
    experiment = config["experiment"]
    requested = experiment.get("run_id")
    if requested:
        return _safe_name(str(requested))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_safe_name(experiment['name'])}"


def stage1_parse(
    config: dict[str, Any],
    run_tag: str | None = None,
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> Path:
    """Parse a raw log once and return the annotated-Parquet artifact."""
    return _stage1_artifacts(
        config, workspace_root=workspace_root, code_root=code_root
    )["annotated"]


def stage2_enrich(
    config: dict[str, Any],
    run_tag: str | None = None,
    *,
    source_run_tag: str | None = None,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> Path:
    """Create (or reuse) templates enriched by the configured LLM."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    stage1 = _stage1_artifacts(config, workspace_root=workspace, code_root=code)
    store = ArtifactStore(workspace, dataset, code)
    ablation = config["ablation"]
    stage_config = {
        "dataset": dataset,
        "llm_enrichment_enabled": bool(ablation["llm_enrichment_enabled"]),
        "enrichment_model_size": "large" if ablation["llm_enrichment_enabled"] else ablation["enrichment_model_size"],
        "enrichment_backend": "deepseek-v4-pro" if ablation["llm_enrichment_enabled"] else None,
        "enrichment_prompt_version": ENRICHMENT_PROMPT_VERSION,
        "enrichment_deployment": (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO")
            if ablation["llm_enrichment_enabled"]
            else None
        ),
        "enrichment_decoding": {"temperature": 0, "max_tokens": None, "max_retries": 2},
    }

    def build(temp_dir: Path) -> dict[str, Path]:
        destination = temp_dir / "templates.json"
        with stage1["templates"].open() as handle:
            templates = json.load(handle)
        if ablation["llm_enrichment_enabled"]:
            enrich_templates(templates, dataset, model_size="large")
        destination.write_text(json.dumps(templates, indent=2))
        provenance = enrichment_provenance(
            templates,
            dataset=dataset,
            enabled=bool(ablation["llm_enrichment_enabled"]),
            deployment=stage_config["enrichment_deployment"],
        )
        provenance_path = temp_dir / "enrichment_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
        return {"templates": destination, "enrichment_provenance": provenance_path}

    outputs, _, reused = store.stage(
        stage="stage2_enrich",
        stage_config=stage_config,
        inputs=[stage1["templates"]],
        build=build,
    )
    _announce("stage2_enrich", reused)
    return outputs["templates"]


def stage3_sequence(
    config: dict[str, Any],
    run_tag: str | None = None,
    *,
    source_run_tag: str | None = None,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> Path:
    """Build (or reuse) explicit HDFS blocks or BGL time-window sequences."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    stage1 = _stage1_artifacts(config, workspace_root=workspace, code_root=code)
    store = ArtifactStore(workspace, dataset, code)
    sequencing = config.get("sequencing", {}).get(dataset, {})
    split_protocol = split_protocol_from_config(config)
    stage_config = {
        "dataset": dataset,
        **sequencing,
        "split_protocol": split_protocol,
        "fit_on": fit_on_from_config(config),
    }

    def build(temp_dir: Path) -> dict[str, Path]:
        frame = _read_parquet(stage1["annotated"])
        kwargs = dict(sequencing)
        kwargs["split"] = split_protocol
        sequences = build_sequences(frame, dataset, **kwargs)
        if not sequences:
            raise ValueError(f"No {dataset.upper()} sequences were built from {stage1['annotated']}")
        flat_sequences = pd.concat(sequences.values(), ignore_index=True)
        destination = temp_dir / "sequences.parquet"
        save_sequences(flat_sequences, destination)
        return {"sequences": destination}

    outputs, _, reused = store.stage(
        stage="stage3_sequence",
        stage_config=stage_config,
        inputs=[stage1["annotated"]],
        build=build,
    )
    _announce("stage3_sequence", reused)
    return outputs["sequences"]


def stage45_build_dataset(
    config: dict[str, Any],
    run_tag: str | None,
    templates_path: str | Path,
    sequences_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
    split_lock: str | Path | None = None,
) -> Path:
    """Build the PyG graph bundle from explicit template and sequence artifacts."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    templates_path = Path(templates_path).resolve()
    sequences_path = Path(sequences_path).resolve()
    labels_path = _hdfs_labels_path(workspace, config)
    inputs = [templates_path, sequences_path]
    if dataset == "hdfs":
        inputs.append(labels_path)
    lock_path = Path(split_lock).resolve() if split_lock else None
    if lock_path is not None:
        inputs.append(lock_path)
    ablation = config["ablation"]
    provenance_path = templates_path.with_name("enrichment_provenance.json")
    if bool(ablation["llm_enrichment_enabled"]):
        if not provenance_path.exists():
            raise FileNotFoundError(
                "LLM graph construction requires frozen enrichment_provenance.json."
            )
        inputs.append(provenance_path)
    identity = graph_identity_from_config(config)
    stage_config = _stage45_config(config)
    if lock_path is not None:
        stage_config["split_lock_id"] = load_split_lock(lock_path)["split_lock_id"]
    store = ArtifactStore(workspace, dataset, code)

    def build(temp_dir: Path) -> dict[str, Path]:
        preferred = str(ablation.get("enrichment_model_size") or "large")
        templates, _, cluster_to_enriched = load_enriched_templates(
            templates_path,
            preferred_size=preferred if preferred in {"large", "small"} else "large",
        )
        if bool(ablation["llm_enrichment_enabled"]):
            require_complete_enrichment(templates, field="enriched_large")
            require_valid_enrichment_provenance(
                templates,
                json.loads(provenance_path.read_text()),
                dataset=dataset,
            )
        if not bool(ablation["llm_enrichment_enabled"]):
            cluster_to_enriched = {}
        structures, unique_stats = _ensure_graph_structures(
            config,
            sequences_path,
            labels_path,
            store=store,
        )
        sequence_ids = [str(item["seq_id"]) for item in structures]
        unit_split_path = _stage1_artifacts(config, workspace_root=workspace, code_root=code).get(
            "unit_split"
        )
        if lock_path is not None:
            idx_train, idx_val, idx_test = apply_split_lock(
                structures,
                load_split_lock(lock_path),
                sequence_ids=sequence_ids,
            )
            lock_id = load_split_lock(lock_path)["split_lock_id"]
        elif split_protocol_from_config(config) == "time":
            idx_train, idx_val, idx_test = split_indices_from_bgl_windows(sequence_ids)
            lock_id = save_split_lock(
                temp_dir / "split_lock.npz",
                idx_train,
                idx_val,
                idx_test,
                sequence_ids=sequence_ids,
                seed=int(config["experiment"]["seed"]),
                protocol=split_protocol_from_config(config),
            )
        elif fit_on_from_config(config) == "train_only" and unit_split_path is not None:
            unit_split = json.loads(Path(unit_split_path).read_text())
            if unit_split.get("kind") == "hdfs_blocks":
                idx_train, idx_val, idx_test = indices_from_unit_split(sequence_ids, unit_split)
                lock_id = save_split_lock(
                    temp_dir / "split_lock.npz",
                    idx_train,
                    idx_val,
                    idx_test,
                    sequence_ids=sequence_ids,
                    seed=int(config["experiment"]["seed"]),
                    protocol=split_protocol_from_config(config),
                )
            else:
                idx_train = idx_val = idx_test = lock_id = None
        else:
            idx_train = idx_val = idx_test = lock_id = None

        if idx_train is None:
            ys = np.array([int(item["y"]) for item in structures])
            idx_train, idx_val, idx_test = split_label_indices(
                ys, seed=int(config["experiment"]["seed"])
            )
            lock_id = save_split_lock(
                temp_dir / "split_lock.npz",
                idx_train,
                idx_val,
                idx_test,
                sequence_ids=sequence_ids,
                seed=int(config["experiment"]["seed"]),
                protocol=split_protocol_from_config(config),
            )

        tfidf_fit_texts = None
        if fit_on_from_config(config) == "train_only":
            train_cids: set[int] = set()
            for index in idx_train:
                train_cids.update(int(cid) for cid in structures[int(index)]["cluster_ids"])
            cid_to_template = {int(item["cluster_id"]): item["template"] for item in templates}
            tfidf_fit_texts = [
                cid_to_template[cid] for cid in sorted(train_cids) if cid in cid_to_template
            ]
        embeddings, cluster_ids, _ = compute_embeddings(
            templates,
            cluster_to_enriched,
            tfidf_enabled=bool(ablation["embeddings"]["tfidf_enabled"]),
            sbert_enabled=bool(ablation["embeddings"]["sbert_enabled"]),
            tfidf_fit_texts=tfidf_fit_texts,
            sbert_text=sbert_text_from_config(config),
        )
        cluster_embeddings = {
            cluster_id: embedding
            for cluster_id, embedding in zip(cluster_ids, embeddings, strict=True)
        }
        print(
            f"[GRAPH] splicing embeddings onto {len(structures)} {dataset} graphs",
            flush=True,
        )
        data_list = attach_cluster_embeddings(
            structures,
            cluster_embeddings,
            use_edge_features=bool(ablation["graph"]["use_edge_features"]),
            include_node_positional_features=bool(
                ablation["graph"].get("node_positional_features", True)
            ),
            include_edge_temporal_features=bool(
                ablation["graph"].get("edge_temporal_features", True)
            ),
            include_edge_positional_features=bool(
                ablation["graph"].get("edge_positional_features", True)
            ),
            missing_embedding="fail",
        )
        if not data_list:
            raise ValueError("Graph builder produced no examples.")
        print(f"[GRAPH] built {len(data_list)} graphs; saving", flush=True)
        sequence_ids = sequence_ids_from_graphs(data_list)
        tfidf_on = bool(ablation["embeddings"]["tfidf_enabled"])
        sbert_on = bool(ablation["embeddings"]["sbert_enabled"])
        tfidf_dim, sbert_dim = embedding_block_dims(
            int(embeddings.shape[1]),
            tfidf_enabled=tfidf_on,
            sbert_enabled=sbert_on,
        )
        node_dim = int(data_list[0].x.shape[1])
        edge_dim = int(data_list[0].edge_attr.shape[1])
        meta = {
            "dataset": dataset,
            "n_total": len(data_list),
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test),
            "anomaly_rate": float(np.mean([item.y.item() for item in data_list])),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "embed_dim": int(embeddings.shape[1]),
            "tfidf_dim": tfidf_dim,
            "sbert_dim": sbert_dim,
            "embedding_flags": ablation["embeddings"],
            "use_edge_features": bool(ablation["graph"]["use_edge_features"]),
            "node_extra_dim": int(node_dim - embeddings.shape[1]),
            "llm_enrichment_enabled": identity["llm_enrichment_enabled"],
            "enrichment_model_size": identity["enrichment_model_size"],
            "feature_contract": identity["feature_contract"],
            "graph_identity": identity,
            "split_lock_id": lock_id,
            "unique_sequences": identity["unique_sequences"],
            "n_raw": unique_stats["n_raw"],
            "n_unique": unique_stats["n_unique"],
            "n_mixed_label_fingerprints": unique_stats["n_mixed_label_fingerprints"],
            "n_graphs_with_oov": int(
                sum(OOV_CLUSTER_ID in set(map(int, item["cluster_ids"])) for item in structures)
            ),
        }
        if bool(ablation["llm_enrichment_enabled"]):
            meta["enrichment_provenance"] = file_digest(provenance_path)
        meta["oov_graph_rate"] = float(meta["n_graphs_with_oov"] / len(structures))
        if unit_split_path is not None:
            unit_meta = json.loads(Path(unit_split_path).read_text())
            for key in ("n_oov_lines", "n_annotated_lines", "oov_line_rate", "protocol"):
                if key in unit_meta:
                    meta[key] = unit_meta[key]
        graph_path = temp_dir / "graph_dataset.pt"
        save_graph_dataset(
            data_list,
            idx_train,
            idx_val,
            idx_test,
            graph_path,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=int(embeddings.shape[1]),
            dataset_meta=meta,
        )
        np.savez_compressed(
            temp_dir / "embeddings.npz",
            cluster_ids=np.asarray(cluster_ids),
            embeddings=embeddings,
        )
        meta_path = temp_dir / "dataset_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        outputs = {
            "graph_dataset": graph_path,
            "embeddings": temp_dir / "embeddings.npz",
            "dataset_meta": meta_path,
        }
        generated_lock = temp_dir / "split_lock.npz"
        if generated_lock.exists():
            outputs["split_lock"] = generated_lock
        return outputs

    outputs, _, reused = store.stage(
        stage="stage45_build_dataset",
        stage_config=stage_config,
        inputs=inputs,
        build=build,
    )
    _announce("stage45_build_dataset", reused)
    return outputs["graph_dataset"]


def stage6_train(
    config: dict[str, Any],
    run_tag: str | None,
    graph_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train the selected GAE variant and return metrics plus training history."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    graph_path = Path(graph_path).resolve()
    training = _resolved_training(config, dataset)
    ablation_graph = config["ablation"]["graph"]
    stage_config = _stage6_config(config)
    store = ArtifactStore(workspace, dataset, code)
    bundle_meta = load_bundle_meta(graph_path) or {}

    def build(temp_dir: Path) -> dict[str, Path]:
        metrics, checkpoint, eval_payload = _train_graph_bundle(
            graph_path,
            training=training,
            seed=int(config["experiment"]["seed"]),
            gine_aggregation=str(ablation_graph["gine_aggregation"]),
            node_transformation=str(ablation_graph["node_transformation"]),
            structure_decoder=structure_decoder_from_config(config),
            node_reconstruct=node_reconstruct_from_config(config),
            fusion_mode=fusion_mode_from_config(config),
            modality_projection_dim=int(config["ablation"]["fusion"].get("modality_projection_dim", 64)),
            node_loss=node_loss_from_config(config),
            bundle_meta=bundle_meta,
        )
        metrics_path = temp_dir / "metrics.json"
        metrics_path.write_text(json.dumps({k: v for k, v in metrics.items() if k != "history_epochs"}, indent=2))
        import torch

        checkpoint_path = temp_dir / "attribute_gae.pt"
        torch.save(checkpoint, checkpoint_path)
        campaign_meta = {
            "campaign_id": config["experiment"].get("campaign_id"),
            "family": config["experiment"].get("family"),
            "experiment_name": config["experiment"].get("name"),
            "run_id": run_tag,
            "graph_identity": graph_identity_from_config(config),
            "split_lock_id": bundle_meta.get("split_lock_id"),
            "parent_graph": _workspace_relative(graph_path, workspace),
            "code_sha": git_revision(code),
            "feature_contract": feature_contract_from_config(config),
            "seed": int(config["experiment"]["seed"]),
            "split_protocol": split_protocol_from_config(config),
            "oov_graph_rate": bundle_meta.get("oov_graph_rate"),
            "oov_line_rate": bundle_meta.get("oov_line_rate"),
        }
        written = write_eval_pack(
            temp_dir,
            metrics={key: value for key, value in metrics.items() if key != "history_epochs"},
            history_epochs=metrics["history_epochs"],
            labels=eval_payload["test_labels"],
            scores=eval_payload["test_scores"],
            structure=eval_payload["structure"],
            node=eval_payload["node"],
            edge=eval_payload["edge"],
            node_blocks={
                name: eval_payload[name] for name in ("tfidf", "sbert", "extras")
            },
            threshold=float(metrics["best_threshold"]),
            alpha=float(training["alpha"]),
            beta=float(training["beta"]),
            gamma=float(training["gamma"]),
            graph_ids=eval_payload["graph_ids"],
            config=config,
            campaign_meta=campaign_meta,
        )
        outputs: dict[str, Path] = {
            "metrics": metrics_path,
            "checkpoint": checkpoint_path,
            "component_metrics": written["component_metrics"],
            "history": written["history"],
            "scores_csv": written["scores_csv"],
            "manifest": written["manifest"],
        }
        for key, path in written.items():
            if key.startswith("figure") or key in {
                "training_history",
                "score_distribution",
                "confusion_matrix",
                "pr_roc",
                "component_contribution",
            }:
                outputs[f"figure_{path.stem}"] = path
        return outputs

    outputs, _, reused = store.stage(
        stage="stage6_train",
        stage_config=stage_config,
        inputs=[graph_path],
        build=build,
    )
    _announce("stage6_train", reused)
    metrics = json.loads(outputs["metrics"].read_text())
    metrics["model_path"] = str(outputs["checkpoint"])
    return metrics


def run_experiment(
    config: dict[str, Any],
    *,
    mode: str,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
    graph_dataset: str | Path | None = None,
    input_run_id: str | None = None,
    checkpoint_root: str | Path | None = None,
    split_lock: str | Path | None = None,
) -> dict[str, Any]:
    """Run one configuration and persist a small run manifest."""
    workspace, code = _roots(config, workspace_root, code_root)
    run_tag = get_run_tag(config)
    dataset = _dataset(config)
    started = time.monotonic()

    if mode in {"full", "smoke"}:
        stage1_parse(config, run_tag, workspace_root=workspace, code_root=code)
        _checkpoint_workspace(workspace, checkpoint_root)
        templates = stage2_enrich(
            config, run_tag, workspace_root=workspace, code_root=code
        )
        _checkpoint_workspace(workspace, checkpoint_root)
        sequences = stage3_sequence(
            config, run_tag, workspace_root=workspace, code_root=code
        )
        _checkpoint_workspace(workspace, checkpoint_root)
        graph_path = stage45_build_dataset(
            config,
            run_tag,
            templates,
            sequences,
            workspace_root=workspace,
            code_root=code,
            split_lock=split_lock,
        )
        _checkpoint_workspace(workspace, checkpoint_root)
    elif mode == "train-only":
        graph_path = _resolve_graph_dataset(
            workspace, graph_dataset=graph_dataset, input_run_id=input_run_id
        )
        assert_train_only_compatible(config, bundle_meta=load_bundle_meta(graph_path))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    metrics = stage6_train(
        config,
        run_tag,
        graph_path,
        workspace_root=workspace,
        code_root=code,
    )
    _checkpoint_workspace(workspace, checkpoint_root)
    output_dir = workspace / config["paths"]["outputs_dir"] / dataset / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(metrics["model_path"]).parent
    _publish_run_dir(cache_dir, output_dir)
    metric_output = output_dir / "metrics.json"
    model_output = output_dir / "attribute_gae.pt"

    record = {
        "run_id": run_tag,
        "dataset": dataset,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_minutes": round((time.monotonic() - started) / 60, 2),
        "config": _serialisable_config(config),
        "graph_identity": graph_identity_from_config(config),
        "artifacts": {
            "graph_dataset": _workspace_relative(graph_path, workspace),
            "metrics": _workspace_relative(metric_output, workspace),
            "checkpoint": _workspace_relative(model_output, workspace),
        },
        "metrics": {key: value for key, value in metrics.items() if key != "history"},
        "history": metrics["history"],
    }
    run_manifest = workspace / "artifacts" / "runs" / f"{run_tag}.json"
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.write_text(json.dumps(record, indent=2))
    _checkpoint_workspace(workspace, checkpoint_root)
    return record


def resolve_hdfs_release_source(
    config: dict[str, Any],
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> HdfsReleaseSource:
    """Locate completed HDFS stage artefacts without rebuilding or retraining."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    if dataset != "hdfs":
        raise ValueError("Inference-release export requires an HDFS configuration.")
    store = ArtifactStore(workspace, dataset, code)
    parser_settings = config["parser"][dataset]
    raw_path = workspace / config["paths"]["raw_dir"] / parser_settings["raw_file"]
    parser_config = code / parser_settings["config"]
    labels_path = _hdfs_labels_path(workspace, config)
    stage1 = _require_completed_stage(
        store,
        stage="stage1_parse",
        stage_config={
            "dataset": dataset,
            "parser_config": parser_settings["config"],
            "raw_file": parser_settings["raw_file"],
        },
        inputs=[raw_path, parser_config],
    )
    ablation = config["ablation"]
    stage2 = _require_completed_stage(
        store,
        stage="stage2_enrich",
        stage_config={
            "dataset": dataset,
            "llm_enrichment_enabled": bool(ablation["llm_enrichment_enabled"]),
            "enrichment_model_size": ablation["enrichment_model_size"],
            "enrichment_prompt_version": ENRICHMENT_PROMPT_VERSION,
        },
        inputs=[stage1["templates"]],
    )
    sequencing = config.get("sequencing", {}).get(dataset, {})
    stage3 = _require_completed_stage(
        store,
        stage="stage3_sequence",
        stage_config={"dataset": dataset, **sequencing},
        inputs=[stage1["annotated"]],
    )
    stage45 = _require_completed_stage(
        store,
        stage="stage45_build_dataset",
        stage_config=_stage45_config(config),
        inputs=[stage2["templates"], stage3["sequences"], labels_path],
    )
    stage6 = _require_completed_stage(
        store,
        stage="stage6_train",
        stage_config=_stage6_config(config),
        inputs=[stage45["graph_dataset"]],
    )
    return HdfsReleaseSource(
        checkpoint_path=stage6["checkpoint"],
        parser_state_path=stage1["parser_state"],
        drain_config_path=parser_config,
        embeddings_path=stage45["embeddings"],
        labels_path=labels_path,
        metrics_path=stage6["metrics"],
    )


def export_completed_hdfs_release(
    config: dict[str, Any],
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
    output_dir: str | Path,
    model_identifier: str,
    version: str,
    bundle_identifier: str | None = None,
    bundle_version: str | None = None,
    pipeline_run_id: str | None = None,
    source: HdfsReleaseSource | None = None,
) -> dict[str, Any]:
    """Export a completed HDFS training run into v2 model and bundle ZIP archives."""
    workspace, code = _roots(config, workspace_root, code_root)
    resolved_source = source or resolve_hdfs_release_source(
        config, workspace_root=workspace, code_root=code
    )
    identity = HdfsReleaseIdentity(
        model_identifier=model_identifier,
        version=version,
        bundle_identifier=bundle_identifier or f"{model_identifier}-preprocessing",
        bundle_version=bundle_version or version,
        pipeline_run_id=pipeline_run_id or config.get("experiment", {}).get("run_id"),
    )
    exported = export_hdfs_inference_release(
        config=config,
        source=resolved_source,
        identity=identity,
        output_dir=Path(output_dir),
        code_root=code,
    )
    return {
        "model_package": str(exported.model_package_path),
        "preprocessing_bundle": str(exported.preprocessing_bundle_path),
        "model_identifier": identity.model_identifier,
        "version": identity.version,
        "bundle_identifier": identity.bundle_identifier,
        "bundle_version": identity.bundle_version,
        "bundle_digest": exported.bundle_manifest.digest,
    }


def _explicit_release_source(args: argparse.Namespace) -> HdfsReleaseSource | None:
    supplied = {
        "checkpoint": args.checkpoint,
        "parser_state": args.parser_state,
        "drain_config": args.drain_config,
        "embeddings": args.embeddings,
        "labels": args.labels,
        "metrics": args.metrics,
    }
    present = {name: path for name, path in supplied.items() if path is not None}
    if not present:
        return None
    missing = sorted(name for name, path in supplied.items() if path is None)
    if missing:
        raise ValueError(
            "export-inference-release explicit artefact flags must be supplied together; "
            f"missing: {', '.join(missing)}"
        )
    return HdfsReleaseSource(
        checkpoint_path=args.checkpoint,
        parser_state_path=args.parser_state,
        drain_config_path=args.drain_config,
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        metrics_path=args.metrics,
    )


def _require_completed_stage(
    store: ArtifactStore,
    *,
    stage: str,
    stage_config: dict[str, Any],
    inputs: list[Path],
) -> dict[str, Path]:
    metadata = input_metadata(inputs, workspace_root=store.workspace_root)
    stage_fingerprint = fingerprint(
        config=stage_config, inputs=metadata, revision=store.revision
    )
    final_dir = store.cache_dir(stage, stage_fingerprint)
    manifest = ArtifactStore._read_valid_manifest(final_dir / SUCCESS_FILE, stage_fingerprint)
    if manifest is None:
        raise FileNotFoundError(
            f"Completed {stage} artefacts are required before exporting an inference release."
        )
    return ArtifactStore._outputs_from_manifest(final_dir, manifest)


def _stage1_artifacts(
    config: dict[str, Any],
    *,
    workspace_root: str | Path | None = None,
    code_root: str | Path | None = None,
) -> dict[str, Path]:
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    parser_settings = config["parser"][dataset]
    raw_path = workspace / config["paths"]["raw_dir"] / parser_settings["raw_file"]
    parser_config = code / parser_settings["config"]
    store = ArtifactStore(workspace, dataset, code)
    fit_on = fit_on_from_config(config)
    labels_path = _hdfs_labels_path(workspace, config) if dataset == "hdfs" else None
    stage_config = {
        "dataset": dataset,
        "parser_config": parser_settings["config"],
        "raw_file": parser_settings["raw_file"],
        "fit_on": fit_on,
        "seed": int(config["experiment"]["seed"]),
        "split_protocol": split_protocol_from_config(config),
    }
    if dataset == "bgl":
        stage_config["sequencing"] = config.get("sequencing", {}).get("bgl", {})
    inputs = [raw_path, parser_config]
    if dataset == "hdfs" and fit_on == "train_only":
        inputs.append(labels_path)

    def build(temp_dir: Path) -> dict[str, Path]:
        parser_cls = BGLParser if dataset == "bgl" else DrainParser
        state_path = temp_dir / "drain_parser.bin"
        parser = parser_cls(
            config_path=str(parser_config),
            persistence_path=str(state_path),
        )
        include_line = None
        unit_payload: dict[str, Any] | None = None
        unmatched = "skip"
        if fit_on == "train_only":
            unmatched = "oov"
            if dataset == "hdfs":
                if labels_path is None or not labels_path.exists():
                    raise FileNotFoundError(
                        "HDFS train_only Drain fit requires anomaly_label.csv."
                    )
                block_ids = scan_hdfs_block_ids(raw_path)
                mapping = _hdfs_label_mapping(labels_path)
                split_protocol = split_protocol_from_config(config)
                if split_protocol == "topology_grouped":
                    # The provisional parser is used only to define groups. The
                    # final parser below is fresh and sees grouped-train lines only.
                    provisional = DrainParser(config_path=str(parser_config))
                    provisional.fit_file(str(raw_path))
                    provisional_frame = provisional.annotate_file(
                        str(raw_path), unmatched="skip"
                    )
                    unit_split = topology_grouped_id_split(
                        block_ids,
                        mapping,
                        hdfs_topology_fingerprints(provisional_frame),
                        seed=int(config["experiment"]["seed"]),
                    )
                else:
                    unit_split = stratified_id_split(
                        block_ids,
                        mapping,
                        seed=int(config["experiment"]["seed"]),
                    )
                unit_payload = {
                    "kind": "hdfs_blocks",
                    "protocol": split_protocol,
                    **unit_split,
                }
                include_line = hdfs_train_line_filter({str(item) for item in unit_payload["train"]})
            else:
                sequencing = config.get("sequencing", {}).get("bgl", {})
                train_cut, val_cut = bgl_time_split_boundaries(
                    raw_path,
                    train_ratio=float(sequencing.get("train_ratio", 0.70)),
                    val_ratio=float(sequencing.get("val_ratio", 0.15)),
                )
                embargo_seconds = int(sequencing.get("embargo_minutes", 0)) * 60
                train_fit_end = train_cut - embargo_seconds
                unit_payload = {
                    "kind": "bgl_time",
                    "protocol": "time",
                    "train_cut_timestamp": train_cut,
                    "val_cut_timestamp": val_cut,
                    "embargo_minutes": int(sequencing.get("embargo_minutes", 0)),
                    "train_fit_end_timestamp": train_fit_end,
                }
                include_line = bgl_train_time_filter(train_fit_end)
        parser.fit_file(str(raw_path), include_line=include_line)
        frame = parser.annotate_file(str(raw_path), unmatched=unmatched)
        templates_path = temp_dir / "templates.json"
        parser.export_templates(str(templates_path))
        n_oov = 0
        if "cluster_id" in frame.columns:
            n_oov = int((frame["cluster_id"] == OOV_CLUSTER_ID).sum())
        if n_oov:
            with templates_path.open() as handle:
                records = json.load(handle)
            records.append(
                {
                    "cluster_id": OOV_CLUSTER_ID,
                    "template": OOV_TEMPLATE,
                    "count": n_oov,
                    "examples": [],
                }
            )
            templates_path.write_text(json.dumps(records, indent=2))
        parser.save()
        annotated_path = temp_dir / "annotated.parquet"
        _write_parquet(frame, annotated_path)
        outputs = {
            "annotated": annotated_path,
            "templates": templates_path,
            "parser_state": state_path,
        }
        if unit_payload is not None:
            unit_payload["n_oov_lines"] = n_oov
            unit_payload["n_annotated_lines"] = int(len(frame))
            unit_payload["oov_line_rate"] = float(n_oov / len(frame)) if len(frame) else 0.0
            unit_path = temp_dir / "unit_split.json"
            save_unit_split(unit_path, unit_payload)
            outputs["unit_split"] = unit_path
        return outputs

    outputs, _, reused = store.stage(
        stage="stage1_parse",
        stage_config=stage_config,
        inputs=[path for path in inputs if path is not None],
        build=build,
    )
    _announce("stage1_parse", reused)
    return outputs


def _train_graph_bundle(
    graph_path: Path,
    *,
    training: dict[str, Any],
    seed: int,
    gine_aggregation: str,
    node_transformation: str,
    structure_decoder: str,
    node_reconstruct: str = "all",
    fusion_mode: str = "concat",
    modality_projection_dim: int = 64,
    node_loss: tuple[str, dict[str, float]] = ("global", {}),
    bundle_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
    )
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader

    from src.modules.models.gae import (
        AttributeAwareGAE,
        compute_anomaly_scores,
        train_epoch,
    )

    seed_everything(seed)
    device = get_device()
    train_graphs, val_graphs, test_graphs, split_meta = load_graph_splits(graph_path)
    node_dim = int(split_meta.get("node_dim") or 0)
    edge_dim = int(split_meta.get("edge_dim") or 0)
    if not node_dim or not edge_dim:
        import torch

        bundle = torch.load(graph_path, weights_only=False, map_location="cpu")
        node_dim = int(bundle["node_dim"])
        edge_dim = int(bundle["edge_dim"])
    if training["train_mode"] == "clean":
        train_graphs = [graph for graph in train_graphs if graph.y.item() == 0]
    if not train_graphs:
        raise ValueError("No training graphs remain after applying train_mode.")

    if training["test_run"]:
        sample_count = int(training["test_samples"])
        train_graphs = train_graphs[:sample_count]
        val_graphs = val_graphs[:sample_count]
        test_graphs = test_graphs[:sample_count]

    edge_mean, edge_std = _normalise_edge_attributes(
        train_graphs,
        val_graphs,
        test_graphs,
        enabled=bool(training["pre_normalize_edges"]),
        minimum_std=float(training["minimum_edge_std"]),
    )
    batch_size = int(training["batch_size"])
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    model = AttributeAwareGAE(
        node_dim=int(node_dim),
        edge_dim=int(edge_dim),
        hidden_dim=int(training["hidden_dim"]),
        latent_dim=int(training["latent_dim"]),
        gine_aggregation=gine_aggregation,
        node_transformation=node_transformation,
        structure_decoder=structure_decoder,
        node_recon_index=_node_recon_index_for_bundle(
            node_dim=int(node_dim),
            node_reconstruct=node_reconstruct,
            split_meta=split_meta,
            bundle_meta=bundle_meta,
        ),
        tfidf_dim=int(bundle_meta.get("tfidf_dim") or split_meta.get("tfidf_dim") or 0),
        sbert_dim=int(bundle_meta.get("sbert_dim") or split_meta.get("sbert_dim") or 0),
        fusion_mode=fusion_mode,
        modality_projection_dim=modality_projection_dim,
        node_loss_mode=node_loss[0],
        node_block_weights=node_loss[1],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    loss_args = {
        "alpha": float(training["alpha"]),
        "beta": float(training["beta"]),
        "gamma": float(training["gamma"]),
    }
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Any] | None = None
    best_val_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(1, int(training["epochs"]) + 1):
        total, structure, node, edge = train_epoch(
            model, train_loader, optimizer, device, **loss_args
        )
        val_scores, val_labels = compute_anomaly_scores(model, val_loader, device, **loss_args)
        threshold, val_metrics = _threshold_and_metrics(val_labels, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": total,
                "train_structure_loss": structure,
                "train_node_loss": node,
                "train_edge_loss": edge,
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )
        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    val_scores, val_labels = compute_anomaly_scores(model, val_loader, device, **loss_args)
    _, val_metrics = _threshold_and_metrics(val_labels, val_scores, threshold=best_threshold)
    test_scores, test_labels, test_components, test_ids = compute_anomaly_scores(
        model, test_loader, device, return_components=True, **loss_args
    )
    test_metrics = _score_metrics(test_labels, test_scores, best_threshold)
    test_predictions = (test_scores > best_threshold).astype(int)
    matrix = confusion_matrix(test_labels, test_predictions, labels=[0, 1]).tolist()
    test_metrics.update(
        {
            "precision": _rounded(precision_score(test_labels, test_predictions, zero_division=0)),
            "recall": _rounded(recall_score(test_labels, test_predictions, zero_division=0)),
            "confusion_matrix": matrix,
        }
    )
    metrics: dict[str, Any] = {
        "best_threshold": _rounded(best_threshold),
        "val_f1": val_metrics["f1"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_f1": test_metrics["f1"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "device": str(device),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_test": len(test_graphs),
        "history": history,
        "history_epochs": history,
    }
    checkpoint = {
        "model_state_dict": best_state,
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
        "hidden_dim": int(training["hidden_dim"]),
        "latent_dim": int(training["latent_dim"]),
        "best_threshold": best_threshold,
        "training": training,
        "history": _loss_history(history),
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "gine_aggregation": gine_aggregation,
        "node_transformation": node_transformation,
        "structure_decoder": structure_decoder,
        "node_reconstruct": node_reconstruct,
        "node_recon_dim": int(model.node_recon_index.numel()),
        "node_recon_index": model.node_recon_index.detach().cpu().tolist(),
        "tfidf_dim": model.tfidf_dim,
        "sbert_dim": model.sbert_dim,
        "fusion_mode": fusion_mode,
        "modality_projection_dim": modality_projection_dim,
        "node_loss_mode": node_loss[0],
        "node_block_weights": node_loss[1],
    }
    metrics["history"] = _loss_history(history)
    eval_payload = {
        "test_scores": test_scores,
        "test_labels": test_labels,
        "structure": test_components["structure"],
        "node": test_components["node"],
        "edge": test_components["edge"],
        "tfidf": test_components["tfidf"],
        "sbert": test_components["sbert"],
        "extras": test_components["extras"],
        "graph_ids": test_ids,
    }
    return metrics, checkpoint, eval_payload


def _train_isolation_forest_bundle(
    graph_path: Path, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the pre-registered non-GNN baseline on frozen BGL graph splits."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    train_graphs, val_graphs, test_graphs, _ = load_graph_splits(graph_path)
    clean_train = [graph for graph in train_graphs if int(graph.y.item()) == 0]
    if not clean_train:
        raise ValueError("No normal training windows remain for Isolation Forest.")
    x_train, x_val, x_test = window_feature_matrices(clean_train, val_graphs, test_graphs)
    model = IsolationForest(
        n_estimators=300,
        max_samples=min(256, len(x_train)),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train)
    # sklearn's score_samples is higher for inliers; negate it for anomaly scoring.
    val_scores = -model.score_samples(x_val)
    val_labels = np.asarray([int(graph.y.item()) for graph in val_graphs])
    threshold, val_metrics = _threshold_and_metrics(val_labels, val_scores)
    test_scores = -model.score_samples(x_test)
    test_labels = np.asarray([int(graph.y.item()) for graph in test_graphs])
    test_metrics = _score_metrics(test_labels, test_scores, threshold)
    predictions = (test_scores > threshold).astype(int)
    test_metrics.update(
        {
            "precision": _rounded(precision_score(test_labels, predictions, zero_division=0)),
            "recall": _rounded(recall_score(test_labels, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist(),
        }
    )
    metrics = {
        "best_threshold": _rounded(threshold),
        "val_f1": val_metrics["f1"], "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
        "test_f1": test_metrics["f1"], "test_pr_auc": test_metrics["pr_auc"], "test_roc_auc": test_metrics["roc_auc"],
        "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "n_train": len(clean_train), "n_val": len(val_graphs), "n_test": len(test_graphs),
        "baseline": "isolation_forest", "n_estimators": 300, "max_samples": min(256, len(x_train)),
    }
    payload = {
        "test_scores": test_scores, "test_labels": test_labels,
        "graph_ids": sequence_ids_from_graphs(test_graphs),
    }
    return metrics, payload


def _node_recon_index_for_bundle(
    *,
    node_dim: int,
    node_reconstruct: str,
    split_meta: dict[str, Any],
    bundle_meta: dict[str, Any] | None,
) -> Any:
    """Full-vector decoder unless fusion asks to skip the SBERT block."""
    import torch

    from src.modules.models.gae import node_recon_index_tensor

    if node_reconstruct != "without_sbert":
        return torch.arange(node_dim, dtype=torch.long)
    meta = dict(bundle_meta or {})
    nested = split_meta.get("dataset_meta")
    if isinstance(nested, dict):
        for key, value in nested.items():
            meta.setdefault(key, value)
    for key, value in split_meta.items():
        meta.setdefault(key, value)
    sbert_dim = sbert_dim_from_meta(meta, node_dim=node_dim)
    return node_recon_index_tensor(
        node_dim,
        sbert_dim=sbert_dim,
        extra_dim=int(meta.get("node_extra_dim", 9)),
    )


def _normalise_edge_attributes(
    train_graphs: list,
    val_graphs: list,
    test_graphs: list,
    *,
    enabled: bool,
    minimum_std: float,
) -> tuple[list[float] | None, list[float] | None]:
    """Fit edge scaling on training graphs only, then apply to every split."""
    if not enabled:
        return None, None
    import torch

    training_edges = [
        graph.edge_attr.float()
        for graph in train_graphs
        if getattr(graph, "edge_attr", None) is not None and graph.edge_attr.numel() > 0
    ]
    if not training_edges:
        return None, None
    stacked = torch.cat(training_edges, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(minimum_std)
    for graph in [*train_graphs, *val_graphs, *test_graphs]:
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel() > 0:
            graph.edge_attr = (edge_attr.float() - mean) / std
    return mean.tolist(), std.tolist()


def _loss_history(history: list[dict[str, float | int]]) -> dict[str, list[float]]:
    """Convert per-epoch records to the compact plot/checkpoint contract."""
    return {
        "total": [float(entry["train_total_loss"]) for entry in history],
        "structure": [float(entry["train_structure_loss"]) for entry in history],
        "node": [float(entry["train_node_loss"]) for entry in history],
        "edge": [float(entry["train_edge_loss"]) for entry in history],
    }


def _threshold_and_metrics(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float | None = None
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import precision_recall_curve

    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        if len(thresholds) == 0:
            threshold = float(np.median(scores))
        else:
            f1_values = (2 * precision[:-1] * recall[:-1]) / (
                precision[:-1] + recall[:-1] + 1e-12
            )
            threshold = float(thresholds[int(np.nanargmax(f1_values))])
    return threshold, _score_metrics(labels, scores, threshold)


def _score_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    predictions = (scores > threshold).astype(int)
    has_both_classes = len(np.unique(labels)) == 2
    return {
        "f1": _rounded(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": _rounded(average_precision_score(labels, scores)) if has_both_classes else 0.0,
        "roc_auc": _rounded(roc_auc_score(labels, scores)) if has_both_classes else 0.0,
    }


def _sequence_labels(
    *,
    dataset: str,
    sequences: dict,
    labels_path: Path | None,
) -> dict[Any, int]:
    if dataset == "bgl":
        return {
            sequence_id: int(group["is_anomaly"].fillna(False).astype(bool).any())
            for sequence_id, group in sequences.items()
        }
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(
            "HDFS requires data/raw/hdfs/anomaly_label.csv for reproducible labels."
        )
    mapping = _hdfs_label_mapping(labels_path)
    missing = [sequence_id for sequence_id in sequences if str(sequence_id) not in mapping]
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise KeyError(
            f"{len(missing)} HDFS sequence(s) are missing from {labels_path} "
            f"(e.g. {preview}). Refusing to treat unlabeled blocks as normal."
        )
    return {sequence_id: mapping[str(sequence_id)] for sequence_id in sequences}


def _to_binary_label(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "anomaly", "anomalous"})
    return int(bool(value))


def _hdfs_label_mapping(labels_path: Path) -> dict[str, int]:
    """Load BlockId → {0,1} from LogHub anomaly_label.csv."""
    labels_frame = pd.read_csv(labels_path)
    columns = {column.lower(): column for column in labels_frame.columns}
    id_column = next(
        (columns[name] for name in ("blockid", "block_id") if name in columns), None
    )
    label_column = next(
        (columns[name] for name in ("label", "anomaly", "is_anomaly") if name in columns), None
    )
    if id_column is None or label_column is None:
        raise ValueError(
            f"Expected BlockId and Label columns in {labels_path}; found {list(labels_frame.columns)}"
        )
    return {
        str(row[id_column]): _to_binary_label(row[label_column])
        for _, row in labels_frame.iterrows()
    }


def _read_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine=_parquet_engine())
    if "parameters" in frame.columns:
        frame["parameters"] = frame["parameters"].apply(
            lambda value: json.loads(value) if isinstance(value, str) else (value or [])
        )
    return frame


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    result = frame.copy()
    if "parameters" in result.columns:
        result["parameters"] = result["parameters"].apply(json.dumps)
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype(object)
    result.to_parquet(path, index=False, engine=_parquet_engine())


def _parquet_engine() -> str:
    """Prefer legacy fastparquet artifacts while allowing a clean PyArrow setup."""
    return "fastparquet" if find_spec("fastparquet") else "pyarrow"


def _roots(
    config: dict[str, Any],
    workspace_root: str | Path | None,
    code_root: str | Path | None,
) -> tuple[Path, Path]:
    meta = config.get("__pipeline__", {})
    workspace_value = workspace_root or meta.get("workspace_root") or os.environ.get(
        "PIPELINE_WORKSPACE_ROOT"
    )
    code_value = code_root or meta.get("code_root")
    if not code_value:
        raise ValueError("Unable to infer code root; use --code-root.")
    code = Path(code_value).resolve()
    workspace = Path(workspace_value).resolve() if workspace_value else code
    return workspace, code


def _dataset(config: dict[str, Any]) -> str:
    dataset = str(config["experiment"]["dataset"]).lower()
    if dataset not in {"bgl", "hdfs"}:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose bgl or hdfs.")
    return dataset


def _hdfs_labels_path(workspace: Path, config: dict[str, Any]) -> Path:
    """Resolve LogHub labels next to the HDFS raw log, with a flat-path fallback."""
    raw_dir = workspace / config["paths"]["raw_dir"]
    raw_file = str(config.get("parser", {}).get("hdfs", {}).get("raw_file") or "hdfs/HDFS_full.log")
    sibling = (raw_dir / raw_file).parent / "anomaly_label.csv"
    flat = raw_dir / "anomaly_label.csv"
    if sibling.exists():
        return sibling
    if flat.exists():
        return flat
    return sibling


def _stage45_config(config: dict[str, Any]) -> dict[str, Any]:
    ablation = config["ablation"]
    return {
        "dataset": _dataset(config),
        "seed": config["experiment"]["seed"],
        "embeddings": ablation["embeddings"],
        "use_edge_features": ablation["graph"]["use_edge_features"],
        "node_positional_features": bool(
            ablation["graph"].get("node_positional_features", True)
        ),
        "edge_temporal_features": bool(
            ablation["graph"].get("edge_temporal_features", True)
        ),
        "edge_positional_features": bool(
            ablation["graph"].get("edge_positional_features", True)
        ),
        "llm_enrichment_enabled": bool(ablation["llm_enrichment_enabled"]),
        "enrichment_model_size": ablation.get("enrichment_model_size"),
        "enrichment_backend": "deepseek-v4-pro" if ablation.get("llm_enrichment_enabled") else None,
        "feature_contract": feature_contract_from_config(config),
        "unique_sequences": unique_sequences_from_config(config),
        "fit_on": fit_on_from_config(config),
        "split_protocol": split_protocol_from_config(config),
    }


def _stage45_structure_config(config: dict[str, Any]) -> dict[str, Any]:
    """Knobs that require rebuilding collapsed topology / extras / time-deltas."""
    return {
        "dataset": _dataset(config),
        "unique_sequences": unique_sequences_from_config(config),
        "feature_contract": feature_contract_from_config(config),
        "split_protocol": split_protocol_from_config(config),
        "fit_on": fit_on_from_config(config),
    }


def _ensure_graph_structures(
    config: dict[str, Any],
    sequences_path: str | Path,
    labels_path: str | Path,
    *,
    store: ArtifactStore,
) -> tuple[list, dict[str, int]]:
    """Reuse embedding-agnostic graph structures across Family A arms."""
    dataset = _dataset(config)
    sequences_path = Path(sequences_path).resolve()
    inputs = [sequences_path]
    if dataset == "hdfs":
        inputs.append(Path(labels_path).resolve())
    stage_config = _stage45_structure_config(config)
    prefer_ids: set[str] | None = None
    if fit_on_from_config(config) == "train_only":
        stage1 = _stage1_artifacts(
            config,
            workspace_root=store.workspace_root,
            code_root=store.code_root,
        )
        unit_split_path = stage1.get("unit_split")
        if unit_split_path is not None:
            inputs.append(Path(unit_split_path).resolve())
            unit_split = json.loads(Path(unit_split_path).read_text())
            prefer_ids = {str(item) for item in unit_split.get("train", [])}
    built: dict[str, tuple[list, dict[str, Any]]] = {}

    def build(temp_dir: Path) -> dict[str, Path]:
        _, sequences = load_sequences(sequences_path, dataset)
        labels = _sequence_labels(
            dataset=dataset,
            sequences=sequences,
            labels_path=labels_path if dataset == "hdfs" else None,
        )
        unique_stats = {
            "n_raw": len(sequences),
            "n_unique": len(sequences),
            "n_mixed_label_fingerprints": 0,
        }
        if unique_sequences_from_config(config):
            sequences, labels, unique_stats = select_unique_sequences(
                sequences, labels, prefer_ids=prefer_ids
            )
            print(
                f"[GRAPH] unique sequences: {unique_stats['n_unique']} / "
                f"{unique_stats['n_raw']} "
                f"({unique_stats['n_mixed_label_fingerprints']} mixed-label fingerprints)",
                flush=True,
            )
        print(
            f"[GRAPH] building {len(sequences)} {dataset} structures "
            f"(contract={stage_config['feature_contract']})",
            flush=True,
        )
        structures, stats = build_graph_structures(
            sequences,
            labels,
            dataset=dataset,
            hdfs_feature_contract=feature_contract_from_config(config),  # type: ignore[arg-type]
            on_graph_error="fail",
        )
        if not structures:
            raise ValueError("Graph builder produced no examples.")
        meta = {
            **unique_stats,
            **stats,
            "dataset": dataset,
            "feature_contract": stage_config["feature_contract"],
            "unique_sequences": stage_config["unique_sequences"],
        }
        structure_path = temp_dir / "graph_structure.pkl"
        save_graph_structures(structure_path, structures, meta)
        built["payload"] = (structures, meta)
        meta_path = temp_dir / "structure_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return {"graph_structure": structure_path, "structure_meta": meta_path}

    outputs, _, reused = store.stage(
        stage="stage45_graph_structure",
        stage_config=stage_config,
        inputs=inputs,
        build=build,
    )
    _announce("stage45_graph_structure", reused)
    memo_key = str(outputs["graph_structure"].resolve())
    cached = _GRAPH_STRUCTURE_MEMO.get(memo_key)
    if cached is None and "payload" in built:
        cached = built["payload"]
        _GRAPH_STRUCTURE_MEMO[memo_key] = cached
    if cached is None:
        cached = load_graph_structures(outputs["graph_structure"])
        _GRAPH_STRUCTURE_MEMO[memo_key] = cached
        print(
            f"[GRAPH] loaded {len(cached[0])} cached structures "
            f"(contract={stage_config['feature_contract']})",
            flush=True,
        )
    elif reused:
        print(
            f"[GRAPH] reusing in-memory structures "
            f"({len(cached[0])} graphs, contract={stage_config['feature_contract']})",
            flush=True,
        )
    structures, meta = cached
    unique_stats = {
        "n_raw": int(meta.get("n_raw", len(structures))),
        "n_unique": int(meta.get("n_unique", len(structures))),
        "n_mixed_label_fingerprints": int(meta.get("n_mixed_label_fingerprints", 0)),
    }
    return structures, unique_stats


def _stage6_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = _dataset(config)
    ablation_graph = config["ablation"]["graph"]
    return {
        "dataset": dataset,
        "seed": config["experiment"]["seed"],
        "training": _resolved_training(config, dataset),
        "gine_aggregation": ablation_graph["gine_aggregation"],
        "node_transformation": ablation_graph["node_transformation"],
        "structure_decoder": structure_decoder_from_config(config),
        "node_reconstruct": node_reconstruct_from_config(config),
        "fusion_mode": fusion_mode_from_config(config),
        "modality_projection_dim": int(config["ablation"]["fusion"].get("modality_projection_dim", 64)),
        "node_loss": node_loss_from_config(config),
    }


def _publish_run_dir(source: Path, output_dir: Path) -> None:
    """Copy a stage-6 cache directory into the public run folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == SUCCESS_FILE:
            continue
        destination = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _resolved_training(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    training = copy.deepcopy(config["training"])
    learning_rate = training.get("learning_rate")
    if isinstance(learning_rate, dict):
        training["learning_rate"] = learning_rate[dataset]
    return training


def _copy_graph_splits(source_graph: Path, dest_graph: Path) -> None:
    """Copy optional per-split shards next to an unpacked graph_dataset.pt."""
    source_dir = source_graph.parent / "graph_dataset_splits"
    if not source_dir.exists():
        source_dir = graph_split_dir(source_graph)
    if not source_dir.exists():
        return
    destination = graph_split_dir(dest_graph)
    if destination.resolve() == source_dir.resolve():
        return
    shutil.copytree(source_dir, destination, dirs_exist_ok=True)


def _resolve_graph_dataset(
    workspace: Path,
    *,
    graph_dataset: str | Path | None,
    input_run_id: str | None,
) -> Path:
    if graph_dataset:
        path = Path(graph_dataset).expanduser().resolve()
        sidecar = path.parent / "dataset_meta.json"
        if path.suffix == ".gz":
            unpacked = gunzip_file(path)
            if sidecar.exists():
                shutil.copy2(sidecar, unpacked.with_name("dataset_meta.json"))
            _copy_graph_splits(path, unpacked)
            path = unpacked
    elif input_run_id:
        manifest = workspace / "artifacts" / "runs" / f"{_safe_name(input_run_id)}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"No prior run manifest: {manifest}")
        path = workspace / json.loads(manifest.read_text())["artifacts"]["graph_dataset"]
    else:
        raise ValueError("train-only mode requires --graph-dataset or --input-run-id.")
    if path.suffix == ".gz":
        unpacked = gunzip_file(path)
        _copy_graph_splits(path, unpacked)
        path = unpacked
    if not path.exists():
        raise FileNotFoundError(f"Graph dataset does not exist: {path}")
    return path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    if not cleaned:
        raise ValueError("Run names must contain at least one letter or number.")
    return cleaned.strip(".-")


def _serialisable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "__pipeline__"}


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return str(path.resolve())


def _optional_workspace_relative(
    path: str | Path | None, workspace: Path, fallback: str | None = None
) -> str | None:
    if path is None:
        return fallback
    return _workspace_relative(Path(path), workspace)


def _checkpoint_workspace(workspace: Path, checkpoint_root: str | Path | None) -> None:
    """Copy completed, ignored artifacts to a mounted durable workspace.

    Cache success manifests are copied only after their payloads.  If a Colab
    runtime disconnects during a large graph transfer, the next session sees no
    valid ``_SUCCESS.json`` and will safely resume the local copy.
    """
    if checkpoint_root is None:
        return
    destination_root = Path(checkpoint_root).resolve()
    if destination_root == workspace.resolve():
        return
    for name in ("artifacts", "models", "outputs", "runs", "campaigns"):
        source = workspace / name
        if not source.exists():
            continue
        destination = destination_root / name
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "--exclude",
                    "_SUCCESS.json",
                    f"{source}/",
                    f"{destination}/",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--include",
                    "*/",
                    "--include",
                    "_SUCCESS.json",
                    "--exclude",
                    "*",
                    f"{source}/",
                    f"{destination}/",
                ],
                check=True,
            )
        else:
            # Local fallback for systems without rsync.  It preserves the same
            # ordering guarantee, albeit without partial-file support.
            manifests: list[tuple[Path, Path]] = []
            for item in source.rglob("*"):
                relative = item.relative_to(source)
                target = destination / relative
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif item.name == "_SUCCESS.json":
                    manifests.append((item, target))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            for source_manifest, destination_manifest in manifests:
                destination_manifest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_manifest, destination_manifest)
    print(f"[CHECKPOINT] {destination_root}")


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _announce(stage: str, reused: bool) -> None:
    print(f"[{'REUSE' if reused else 'DONE'}] {stage}")


def _write_matrix_summary(
    workspace: Path, results: list[dict[str, Any]], *, dataset: str
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = workspace / "outputs" / dataset / f"{timestamp}_ablation_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(output_dir / "ablation_results.csv", index=False)
    return json_path


def prepare_representation_campaign(
    config: dict[str, Any],
    *,
    campaign_id: str,
    workspace_root: str | Path,
    code_root: str | Path,
    campaign_dir: str | Path | None = None,
    matrix_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    """Parse/enrich/sequence once, then build gzipped Family A graph bundles."""
    workspace, code = _roots(config, workspace_root, code_root)
    dataset = _dataset(config)
    campaign_id = _safe_name(campaign_id)
    destination = Path(campaign_dir).resolve() if campaign_dir else (
        workspace / "campaigns" / campaign_id
    )
    destination.mkdir(parents=True, exist_ok=True)
    if matrix_path:
        matrix_file = Path(matrix_path).resolve()
    elif dataset == "bgl":
        matrix_file = code / "configs" / "ablation_representation_bgl.yaml"
    else:
        matrix_file = code / "configs" / "ablation_representation.yaml"
    matrix = load_matrix(matrix_file)
    experiments = enabled_experiments(matrix)
    if arms:
        requested = {_safe_name(name) for name in arms}
        available = {_safe_name(str(item["name"])) for item in experiments}
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Unknown Family A arm(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(available))}."
            )
        experiments = [
            item for item in experiments if _safe_name(str(item["name"])) in requested
        ]
    _require_torch_geometric()
    print("[CAMPAIGN] enrichment backend: deepseek-v4-pro")

    def _arm_bundle_complete(arm_name: str) -> bool:
        bundle = destination / "graphs" / arm_name
        return (bundle / "graph_dataset.pt.gz").exists() and (bundle / "dataset_meta.json").exists()

    pending = [
        _safe_name(str(item["name"]))
        for item in experiments
        if not _arm_bundle_complete(_safe_name(str(item["name"])))
    ]
    for item in experiments:
        name = _safe_name(str(item["name"]))
        status = "pending" if name in pending else "reuse"
        print(f"[CAMPAIGN] {name}: {status}")

    templates: Path | None = None
    sequences: Path | None = None
    if pending:
        enrich_config = apply_overrides(config, ["ablation.enrichment_model_size=large"])
        stage1_parse(enrich_config, campaign_id, workspace_root=workspace, code_root=code)
        _checkpoint_workspace(workspace, checkpoint_root)
        templates = stage2_enrich(
            enrich_config, campaign_id, workspace_root=workspace, code_root=code
        )
        _checkpoint_workspace(workspace, checkpoint_root)
        sequences = stage3_sequence(
            enrich_config, campaign_id, workspace_root=workspace, code_root=code
        )
        _checkpoint_workspace(workspace, checkpoint_root)
    else:
        print("[CAMPAIGN] all graph bundles present; skipping parse/enrich/sequence/rebuild")

    split_lock_path = destination / "split_lock.npz"
    graphs: list[dict[str, Any]] = []
    lock_for_build: Path | None = split_lock_path if split_lock_path.exists() else None

    for experiment in experiments:
        name = _safe_name(str(experiment["name"]))
        arm_config = apply_overrides(
            config,
            [f"{key}={json.dumps(value)}" for key, value in experiment["overrides"].items()],
        )
        arm_config["experiment"]["campaign_id"] = campaign_id
        arm_config["experiment"]["family"] = "representation"
        bundle_dir = destination / "graphs" / name
        existing = bundle_dir / "graph_dataset.pt.gz"
        existing_meta = bundle_dir / "dataset_meta.json"
        if existing.exists() and existing_meta.exists():
            graphs.append(
                {
                    "name": name,
                    "graph_dataset": str(existing.relative_to(destination)),
                    "dataset_meta": "graphs/" + name + "/dataset_meta.json",
                    "identity": graph_identity_from_config(arm_config),
                    "node_reconstruct": node_reconstruct_from_config(arm_config),
                    "digest": file_digest(existing),
                    "node_dim": json.loads(existing_meta.read_text()).get("node_dim"),
                    "reused": True,
                }
            )
            print(f"[CAMPAIGN] {name} already present → {existing}")
            continue
        if templates is None or sequences is None:
            raise RuntimeError(
                f"Cannot rebuild {name}: parse/enrich/sequence artifacts are missing."
            )
        graph_path = stage45_build_dataset(
            arm_config,
            f"{campaign_id}_{name}",
            templates,
            sequences,
            workspace_root=workspace,
            code_root=code,
            split_lock=lock_for_build,
        )
        cache_dir = graph_path.parent
        if lock_for_build is None and (cache_dir / "split_lock.npz").exists():
            shutil.copy2(cache_dir / "split_lock.npz", split_lock_path)
            lock_for_build = split_lock_path
        bundle_dir = destination / "graphs" / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        compressed = gzip_file(graph_path, bundle_dir / "graph_dataset.pt.gz")
        meta_source = cache_dir / "dataset_meta.json"
        if meta_source.exists():
            shutil.copy2(meta_source, bundle_dir / "dataset_meta.json")
        split_source = graph_split_dir(graph_path)
        if split_source.exists():
            shutil.copytree(split_source, bundle_dir / "graph_dataset_splits", dirs_exist_ok=True)
        graphs.append(
            {
                "name": name,
                "graph_dataset": str(compressed.relative_to(destination)),
                "dataset_meta": "graphs/" + name + "/dataset_meta.json",
                "identity": graph_identity_from_config(arm_config),
                "node_reconstruct": node_reconstruct_from_config(arm_config),
                "digest": file_digest(compressed),
                "node_dim": json.loads((bundle_dir / "dataset_meta.json").read_text()).get("node_dim")
                if (bundle_dir / "dataset_meta.json").exists()
                else None,
            }
        )
        _checkpoint_workspace(workspace, checkpoint_root)
        print(f"[CAMPAIGN] {name} → {compressed}")

    previous: dict[str, Any] = {}
    previous_manifest = destination / "manifest.json"
    if previous_manifest.exists():
        try:
            loaded = json.loads(previous_manifest.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            previous = loaded
    previous_graphs = {
        str(item.get("name")): item
        for item in previous.get("graphs", [])
        if isinstance(item, dict) and item.get("name")
    }
    # A BGL research campaign is prepared in stages: representation arms first,
    # then PB3 arms. Keep manifest entries for completed arms that are absent
    # from the current matrix, while replacing any arm rebuilt in this pass.
    merged_graphs = {**previous_graphs, **{str(item["name"]): item for item in graphs}}
    extra = {
        "code_sha": git_revision(code),
        "matrix": str(matrix_file),
        "templates": _optional_workspace_relative(templates, workspace, previous.get("templates")),
        "sequences": _optional_workspace_relative(sequences, workspace, previous.get("sequences")),
        "seeds": [int(seed) for seed in (matrix.get("seeds") or [config["experiment"]["seed"]])],
    }
    manifest_path = write_campaign_manifest(
        destination,
        campaign_id=campaign_id,
        dataset=dataset,
        family="representation",
        graphs=[merged_graphs[name] for name in sorted(merged_graphs)],
        split_lock=split_lock_path if split_lock_path.exists() else None,
        extra=extra,
    )
    _checkpoint_workspace(workspace, checkpoint_root)
    print(f"Campaign manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _completed_run_dir(
    workspace: Path,
    dataset: str,
    run_id: str,
    *,
    expected_config: dict[str, Any],
) -> Path | None:
    """Return a compatible completed run, never a stale smoke/config result."""
    metrics = workspace / "outputs" / dataset / run_id / "metrics.json"
    config_path = metrics.parent / "config.yaml"
    if not metrics.exists() or not config_path.exists():
        return None
    try:
        completed_config = yaml.safe_load(config_path.read_text())
        if _stage6_config(completed_config) != _stage6_config(expected_config):
            return None
    except (KeyError, TypeError, yaml.YAMLError):
        return None
    return metrics.parent


def _run_matrix(args: argparse.Namespace, base_config: dict[str, Any]) -> int:
    matrix_path = Path(args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    experiments = enabled_experiments(matrix)
    pairs = experiment_seed_pairs(matrix, int(base_config["experiment"]["seed"]))
    if not pairs:
        raise ValueError(f"No experiments found in {matrix_path}")
    if (
        args.mode == "train-only"
        and bool(matrix.get("requires_graph_rebuild"))
        and not args.campaign_dir
    ):
        raise GraphIdentityError(
            f"{matrix_path.name} is a representation matrix. Prepare graphs with "
            "`--mode prepare-campaign` and train with `--campaign-dir`, or use "
            "configs/ablation_train.yaml with a single baseline graph."
        )
    results: list[dict[str, Any]] = []
    dataset = _dataset(base_config)
    workspace = Path(args.workspace_root).resolve()
    campaign_id = args.campaign_id or args.run_id
    for experiment, seed in pairs:
        name = experiment["name"]
        config = apply_overrides(
            base_config,
            [f"{key}={json.dumps(value)}" for key, value in experiment["overrides"].items()],
        )
        if campaign_id:
            config["experiment"]["campaign_id"] = campaign_id
            config["experiment"]["run_id"] = f"{_safe_name(campaign_id)}_{_safe_name(name)}"
        elif args.run_id:
            config["experiment"]["run_id"] = f"{args.run_id}_{name}"
        config["experiment"]["family"] = matrix.get("family") or config["experiment"].get("family")
        config["experiment"]["name"] = name
        config["experiment"]["seed"] = seed
        if campaign_id:
            config["experiment"]["run_id"] = (
                f"{_safe_name(campaign_id)}_{_safe_name(name)}_seed{seed}"
            )
        run_id = get_run_tag(config)
        existing = _completed_run_dir(
            workspace, dataset, run_id, expected_config=config
        )
        if existing is not None:
            metrics = json.loads((existing / "metrics.json").read_text())
            results.append({"name": name, "seed": seed, "status": "OK", "reused": True, **metrics})
            print(f"[SKIP] {name} already completed at {existing}")
            continue
        graph_dataset = args.graph_dataset
        if experiment_requires_graph_rebuild(experiment, matrix) and args.mode == "train-only":
            if not args.campaign_dir:
                results.append(
                    {
                        "name": name,
                        "status": "FAILED",
                        "error": (
                            f"{name} requires a rebuilt graph. Use Family A campaign "
                            "prepare + --campaign-dir, not a shared --graph-dataset."
                        ),
                    }
                )
                continue
        try:
            record = run_experiment(
                config,
                mode=args.mode,
                workspace_root=args.workspace_root,
                code_root=args.code_root,
                graph_dataset=graph_dataset,
                input_run_id=args.input_run_id,
                checkpoint_root=args.checkpoint_root,
                split_lock=args.split_lock,
            )
            results.append(
                {
                    "name": name,
                    "seed": seed,
                    "status": "OK",
                    **record["metrics"],
                    "history": record["history"],
                }
            )
        except Exception as exc:  # Keep the matrix resumable even after one failed arm.
            results.append(
                {
                    "name": name,
                    "seed": seed,
                    "status": "FAILED",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    summary = _write_matrix_summary(workspace, results, dataset=dataset)
    if campaign_id:
        try:
            write_campaign_report(workspace, dataset=dataset, campaign_id=_safe_name(campaign_id))
        except FileNotFoundError:
            pass
    _checkpoint_workspace(workspace, args.checkpoint_root)
    print(f"Matrix results: {summary}")
    return 0 if all(result["status"] == "OK" for result in results) else 1


def _run_campaign_dir(args: argparse.Namespace, base_config: dict[str, Any]) -> int:
    """Train Family A bundles from a prepared campaign directory, one graph at a time."""
    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = json.loads((campaign_dir / "manifest.json").read_text())
    dataset = str(manifest.get("dataset") or _dataset(base_config))
    campaign_id = str(manifest.get("campaign_id") or args.campaign_id or campaign_dir.name)
    workspace = Path(args.workspace_root).resolve()
    results: list[dict[str, Any]] = []
    staged_uncompressed: Path | None = None
    try:
        seeds = [int(seed) for seed in (manifest.get("seeds") or [base_config["experiment"]["seed"]])]
        for graph_entry in manifest.get("graphs") or []:
            name = graph_entry["name"]
            overrides = [f"experiment.name={name}"]
            identity = graph_entry.get("identity") or {}
            if "llm_enrichment_enabled" in identity:
                overrides.append(
                    f"ablation.llm_enrichment_enabled={json.dumps(identity['llm_enrichment_enabled'])}"
                )
            if identity.get("enrichment_model_size"):
                overrides.append(
                    f"ablation.enrichment_model_size={json.dumps(identity['enrichment_model_size'])}"
                )
            if "tfidf_enabled" in identity:
                overrides.append(
                    f"ablation.embeddings.tfidf_enabled={json.dumps(identity['tfidf_enabled'])}"
                )
            if "sbert_enabled" in identity:
                overrides.append(
                    f"ablation.embeddings.sbert_enabled={json.dumps(identity['sbert_enabled'])}"
                )
            if "use_edge_features" in identity:
                overrides.append(
                    f"ablation.graph.use_edge_features={json.dumps(identity['use_edge_features'])}"
                )
            for key in (
                "node_positional_features",
                "edge_temporal_features",
                "edge_positional_features",
            ):
                if key in identity:
                    overrides.append(
                        f"ablation.graph.{key}={json.dumps(identity[key])}"
                    )
            if "unique_sequences" in identity:
                overrides.append(
                    f"ablation.graph.unique_sequences={json.dumps(identity['unique_sequences'])}"
                )
            if identity.get("fit_on"):
                overrides.append(
                    f"ablation.representation.fit_on={json.dumps(identity['fit_on'])}"
                )
            if identity.get("split_protocol"):
                overrides.append(
                    f"sequencing.{dataset}.split={json.dumps(identity['split_protocol'])}"
                )
            if identity.get("sbert_text"):
                overrides.append(
                    f"ablation.embeddings.sbert_text={json.dumps(identity['sbert_text'])}"
                )
            if graph_entry.get("node_reconstruct"):
                overrides.append(
                    f"ablation.fusion.node_reconstruct={json.dumps(graph_entry['node_reconstruct'])}"
                )
            if identity.get("feature_contract"):
                overrides.append(
                    f"ablation.feature_contract={json.dumps(identity['feature_contract'])}"
                )
            relative = graph_entry["graph_dataset"]
            compressed = campaign_dir / relative
            if not compressed.exists():
                compressed = Path(relative)
            if staged_uncompressed is not None and staged_uncompressed.exists():
                staged_uncompressed.unlink()
            staged_uncompressed = workspace / "data" / "processed" / dataset / (
                f"{_safe_name(campaign_id)}_{_safe_name(name)}_graph_dataset.pt"
            )
            staged_uncompressed = gunzip_file(compressed, staged_uncompressed)
            meta_relative = graph_entry.get("dataset_meta")
            if meta_relative:
                meta_source = campaign_dir / meta_relative
                if meta_source.exists():
                    shutil.copy2(
                        meta_source, staged_uncompressed.with_name("dataset_meta.json")
                    )

            for seed in seeds:
                run_id = f"{_safe_name(campaign_id)}_{_safe_name(name)}_seed{seed}"
                config = apply_overrides(base_config, overrides)
                config["experiment"]["campaign_id"] = campaign_id
                config["experiment"]["family"] = "representation"
                config["experiment"]["run_id"] = run_id
                config["experiment"]["name"] = name
                config["experiment"]["seed"] = seed
                existing = _completed_run_dir(
                    workspace, dataset, run_id, expected_config=config
                )
                if existing is not None:
                    metrics = json.loads((existing / "metrics.json").read_text())
                    results.append({"name": name, "seed": seed, "status": "OK", "reused": True, **metrics})
                    print(f"[SKIP] {name} seed={seed}")
                    continue
                try:
                    record = run_experiment(
                        config,
                        mode="train-only",
                        workspace_root=workspace,
                        code_root=args.code_root,
                        graph_dataset=staged_uncompressed,
                        checkpoint_root=args.checkpoint_root,
                    )
                    results.append({"name": name, "seed": seed, "status": "OK", **record["metrics"], "history": record["history"]})
                except Exception as exc:
                    results.append(
                        {
                            "name": name,
                            "seed": seed,
                            "status": "FAILED",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                _checkpoint_workspace(workspace, args.checkpoint_root)
    finally:
        if staged_uncompressed is not None and staged_uncompressed.exists():
            staged_uncompressed.unlink()
    summary = _write_matrix_summary(workspace, results, dataset=dataset)
    try:
        write_campaign_report(workspace, dataset=dataset, campaign_id=_safe_name(campaign_id))
    except FileNotFoundError:
        pass
    print(f"Campaign train results: {summary}")
    return 0 if all(result["status"] == "OK" for result in results) else 1


def _run_isolation_forest_campaign(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Train the prescribed simple BGL baseline from the frozen hybrid graph."""
    if _dataset(config) != "bgl":
        raise ValueError("Isolation Forest campaign mode is registered for BGL only.")
    if not args.campaign_dir:
        raise ValueError("--campaign-dir is required for --mode isolation-forest.")
    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = json.loads((campaign_dir / "manifest.json").read_text())
    baseline_graph = next(
        (item for item in manifest.get("graphs", []) if item.get("name") == "hybrid_llm"), None
    )
    if baseline_graph is None:
        raise FileNotFoundError("Prepared campaign must contain the hybrid_llm graph bundle.")
    workspace = Path(args.workspace_root).resolve()
    campaign_id = str(manifest.get("campaign_id") or args.campaign_id or campaign_dir.name)
    compressed = campaign_dir / baseline_graph["graph_dataset"]
    graph_path = workspace / "data" / "processed" / "bgl" / f"{_safe_name(campaign_id)}_iforest.pt"
    graph_path = gunzip_file(compressed, graph_path)
    meta_relative = baseline_graph.get("dataset_meta")
    if meta_relative and (campaign_dir / meta_relative).exists():
        shutil.copy2(campaign_dir / meta_relative, graph_path.with_name("dataset_meta.json"))
    seeds = config.get("baseline", {}).get("isolation_forest", {}).get("seeds", [13, 29, 42, 71, 101])
    try:
        for raw_seed in seeds:
            seed = int(raw_seed)
            run_id = f"{_safe_name(campaign_id)}_isolation_forest_seed{seed}"
            output_dir = workspace / config["paths"]["outputs_dir"] / "bgl" / run_id
            if (output_dir / "metrics.json").exists():
                print(f"[SKIP] isolation_forest seed={seed}")
                continue
            metrics, payload = _train_isolation_forest_bundle(graph_path, seed=seed)
            campaign_meta = {
                "campaign_id": campaign_id, "family": "baseline",
                "experiment_name": "isolation_forest", "run_id": run_id,
                "seed": seed, "split_lock_id": (load_bundle_meta(graph_path) or {}).get("split_lock_id"),
                "baseline_features": "train_template_counts,oov_share,log1p_window_length",
            }
            zeros = np.zeros_like(payload["test_scores"], dtype=float)
            write_eval_pack(
                output_dir, metrics=metrics, history_epochs=[], labels=payload["test_labels"],
                scores=payload["test_scores"], structure=zeros, node=zeros, edge=zeros,
                threshold=float(metrics["best_threshold"]), alpha=1.0, beta=1.0, gamma=1.0,
                graph_ids=payload["graph_ids"], config=config, campaign_meta=campaign_meta,
            )
            print(f"[BASELINE] isolation_forest seed={seed} → {output_dir}")
    finally:
        if graph_path.exists():
            graph_path.unlink()
    report = write_campaign_report(workspace, dataset="bgl", campaign_id=_safe_name(campaign_id), baseline_name="hybrid_llm")
    print(f"Isolation Forest campaign report: {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "train-only",
            "smoke",
            "export-inference-release",
            "prepare-campaign",
            "isolation-forest",
            "report",
        ),
        default="full",
        help="Full pipeline, train-only, smoke, release export, Family A prepare, BGL Isolation Forest, or campaign report.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation_base.yaml"),
        help="Base experiment YAML.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="Optional matrix YAML to apply on top of the base config.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.environ.get("PIPELINE_WORKSPACE_ROOT", ".")),
        help="Ignored data/artifact root (separate from the code checkout).",
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository checkout that contains configs/ and src/.",
    )
    parser.add_argument("--run-id", help="Optional explicit id for this run.")
    parser.add_argument(
        "--input-run-id", help="Prior full-run id whose graph bundle should be trained."
    )
    parser.add_argument(
        "--graph-dataset",
        type=Path,
        help="Explicit graph_dataset.pt path for train-only mode.",
    )
    parser.add_argument(
        "--split-lock",
        type=Path,
        help="Frozen idx_train/val/test npz injected into stage45.",
    )
    parser.add_argument("--campaign-id", help="Shared id prefix for a campaign of runs.")
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        help="Prepared campaign folder (manifest.json + graphs/).",
    )
    parser.add_argument(
        "--family",
        choices=("A", "B", "representation", "train"),
        help="A = train each Family A graph in --campaign-dir; B = train-only matrix on one graph.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Mounted durable workspace copied after every successful stage.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="YAML-aware override; may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for export-inference-release ZIP archives.",
    )
    parser.add_argument(
        "--model-identifier",
        default="attribute-gae",
        help="model_identifier written into the exported v2 package.",
    )
    parser.add_argument(
        "--model-version",
        default="v2",
        help="Package/bundle version written into the exported release.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Training checkpoint for export.")
    parser.add_argument("--parser-state", type=Path, help="Frozen Drain snapshot for export.")
    parser.add_argument("--drain-config", type=Path, help="Drain INI used at fit time.")
    parser.add_argument("--embeddings", type=Path, help="Frozen embeddings.npz for export.")
    parser.add_argument("--labels", type=Path, help="HDFS anomaly_label.csv for export.")
    parser.add_argument("--metrics", type=Path, help="Training metrics JSON for export.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = args.code_root / config_path
    config = load_config(config_path)
    config = apply_overrides(config, args.overrides)
    config["__pipeline__"]["workspace_root"] = str(args.workspace_root.resolve())
    config["__pipeline__"]["code_root"] = str(args.code_root.resolve())
    if args.run_id and not args.matrix and args.mode not in {"prepare-campaign", "report"}:
        config["experiment"]["run_id"] = args.run_id
    if args.campaign_id:
        config["experiment"]["campaign_id"] = args.campaign_id
    if args.mode == "prepare-campaign":
        prepared = prepare_representation_campaign(
            config,
            campaign_id=args.campaign_id or args.run_id or get_run_tag(config),
            workspace_root=args.workspace_root,
            code_root=args.code_root,
            campaign_dir=args.campaign_dir,
            matrix_path=args.matrix,
            checkpoint_root=args.checkpoint_root,
        )
        print(json.dumps(prepared, indent=2, default=str))
        return 0
    if args.mode == "report":
        campaign_id = args.campaign_id or args.run_id
        if not campaign_id:
            raise ValueError("--campaign-id is required for --mode report.")
        summary = write_campaign_report(
            args.workspace_root,
            dataset=_dataset(config),
            campaign_id=_safe_name(campaign_id),
        )
        print(f"Campaign report: {summary}")
        return 0
    if args.mode == "isolation-forest":
        return _run_isolation_forest_campaign(args, config)
    family = (args.family or "").lower()
    if args.mode == "smoke":
        config = apply_overrides(
            config,
            [
                "ablation.llm_enrichment_enabled=false",
                "training.test_run=true",
                "training.epochs=1",
            ],
        )
    if args.mode == "export-inference-release":
        if args.matrix:
            raise ValueError("export-inference-release does not accept --matrix.")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(args.workspace_root).resolve() / "releases" / "hdfs"
        explicit_source = _explicit_release_source(args)
        exported = export_completed_hdfs_release(
            config,
            workspace_root=args.workspace_root,
            code_root=args.code_root,
            output_dir=output_dir,
            model_identifier=args.model_identifier,
            version=args.model_version,
            pipeline_run_id=args.run_id or config.get("experiment", {}).get("run_id"),
            source=explicit_source,
        )
        print(json.dumps(exported, indent=2))
        return 0
    if args.mode == "train-only" and args.campaign_dir and family in {"", "a", "representation"}:
        return _run_campaign_dir(args, config)
    if args.mode == "train-only" and args.campaign_dir and family in {"b", "train"} and not args.graph_dataset:
        manifest = json.loads(Path(args.campaign_dir).resolve().joinpath("manifest.json").read_text())
        baseline = next(
            (item for item in manifest.get("graphs", []) if item["name"] == "baseline_full"),
            None,
        )
        if baseline is None:
            raise FileNotFoundError("Campaign has no baseline_full graph for Family B.")
        identity = baseline.get("identity") or {}
        identity_overrides: list[str] = []
        identity_paths = {
            "llm_enrichment_enabled": "ablation.llm_enrichment_enabled",
            "enrichment_model_size": "ablation.enrichment_model_size",
            "tfidf_enabled": "ablation.embeddings.tfidf_enabled",
            "sbert_enabled": "ablation.embeddings.sbert_enabled",
            "use_edge_features": "ablation.graph.use_edge_features",
            "feature_contract": "ablation.feature_contract",
            "unique_sequences": "ablation.graph.unique_sequences",
            "fit_on": "ablation.representation.fit_on",
            "sbert_text": "ablation.embeddings.sbert_text",
        }
        for key, dotted in identity_paths.items():
            if key in identity and identity[key] is not None:
                identity_overrides.append(f"{dotted}={json.dumps(identity[key])}")
        if identity.get("split_protocol"):
            identity_overrides.append(
                f"sequencing.{_dataset(config)}.split={json.dumps(identity['split_protocol'])}"
            )
        config = apply_overrides(config, identity_overrides)
        compressed = Path(args.campaign_dir).resolve() / baseline["graph_dataset"]
        uncompressed = Path(args.workspace_root).resolve() / "data" / "processed" / _dataset(config) / "baseline_full_graph_dataset.pt"
        args.graph_dataset = gunzip_file(compressed, uncompressed)
        meta_relative = baseline.get("dataset_meta")
        if meta_relative:
            meta_source = Path(args.campaign_dir).resolve() / meta_relative
            if meta_source.exists():
                shutil.copy2(meta_source, uncompressed.with_name("dataset_meta.json"))
        if args.matrix is None:
            args.matrix = Path(args.code_root) / "configs" / "ablation_train.yaml"
        if args.campaign_id is None:
            args.campaign_id = manifest.get("campaign_id")
    if args.matrix:
        return _run_matrix(args, config)
    record = run_experiment(
        config,
        mode=args.mode,
        workspace_root=args.workspace_root,
        code_root=args.code_root,
        graph_dataset=args.graph_dataset,
        input_run_id=args.input_run_id,
        checkpoint_root=args.checkpoint_root,
        split_lock=args.split_lock,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
