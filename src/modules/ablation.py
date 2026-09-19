"""HDFS ablation campaign helpers: graph identity, split locks, eval packs.

Family A (representation) rebuilds ``graph_dataset.pt``. Family B (train)
reuses one frozen bundle. Train-only mode must refuse a config whose graph
identity disagrees with the bundle metadata.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

LEGACY_GRAPH_IDENTITY: dict[str, Any] = {
    "llm_enrichment_enabled": True,
    "enrichment_model_size": "large",
    "tfidf_enabled": True,
    "sbert_enabled": True,
    "use_edge_features": True,
    "feature_contract": "notebook_raw_v1",
    "unique_sequences": False,
    "fit_on": "all",
    "split_protocol": "stratified",
    "sbert_text": "embedding_text",
}

GRAPH_IDENTITY_KEYS = tuple(LEGACY_GRAPH_IDENTITY.keys())


class GraphIdentityError(ValueError):
    """Raised when train-only is asked to train a graph it does not match."""


def feature_contract_from_config(config: Mapping[str, Any]) -> str:
    """Return the HDFS feature encoding declared by *config*."""
    ablation = config.get("ablation") or {}
    declared = ablation.get("feature_contract")
    if declared:
        return str(declared)
    release = config.get("inference_release") or {}
    return str(release.get("feature_contract", "stabilized_v2"))


def unique_sequences_from_config(config: Mapping[str, Any]) -> bool:
    """HDFS unique-sequence graphs; always ``False`` for BGL."""
    experiment = config.get("experiment") or {}
    dataset = str(experiment.get("dataset") or "bgl").lower()
    if dataset != "hdfs":
        return False
    graph = (config.get("ablation") or {}).get("graph") or {}
    return bool(graph.get("unique_sequences", True))


def fit_on_from_config(config: Mapping[str, Any]) -> str:
    """``all`` (transductive Drain+TF-IDF) or ``train_only`` (inductive_v1).

    Missing YAML keys stay ``all`` so older configs (notebook parity) do not
    silently change protocol.
    """
    representation = ((config.get("ablation") or {}).get("representation")) or {}
    value = str(representation.get("fit_on") or "all").lower()
    if value not in {"all", "train_only"}:
        raise ValueError(f"ablation.representation.fit_on must be 'all' or 'train_only', got {value!r}")
    return value


def split_protocol_from_config(config: Mapping[str, Any]) -> str:
    """Graph/window split protocol."""
    experiment = config.get("experiment") or {}
    dataset = str(experiment.get("dataset") or "bgl").lower()
    sequencing = (config.get("sequencing") or {}).get(dataset) or {}
    value = str(sequencing.get("split") or "stratified").lower()
    allowed = {"time", "stratified"} if dataset == "bgl" else {"stratified", "topology_grouped"}
    if value not in allowed:
        raise ValueError(
            f"sequencing.{dataset}.split must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def structure_decoder_from_config(config: Mapping[str, Any]) -> str:
    """Directed concat-MLP (default) or symmetric inner product."""
    graph = ((config.get("ablation") or {}).get("graph")) or {}
    value = str(graph.get("structure_decoder") or "mlp").lower()
    if value not in {"mlp", "inner_product"}:
        raise ValueError(
            f"ablation.graph.structure_decoder must be 'mlp' or 'inner_product', got {value!r}"
        )
    return value


def sbert_text_from_config(config: Mapping[str, Any]) -> str:
    """How MiniLM text is assembled. Missing YAML stays ``embedding_text``."""
    embeddings = ((config.get("ablation") or {}).get("embeddings")) or {}
    value = str(embeddings.get("sbert_text") or "embedding_text").lower()
    if value not in {"embedding_text", "grounded_v1"}:
        raise ValueError(
            f"ablation.embeddings.sbert_text must be 'embedding_text' or 'grounded_v1', got {value!r}"
        )
    return value


def node_reconstruct_from_config(config: Mapping[str, Any]) -> str:
    """Node-decoder target: full ``x`` or TF-IDF+extras (skip SBERT)."""
    fusion = ((config.get("ablation") or {}).get("fusion")) or {}
    value = str(fusion.get("node_reconstruct") or "all").lower()
    if value not in {"all", "without_sbert"}:
        raise ValueError(
            f"ablation.fusion.node_reconstruct must be 'all' or 'without_sbert', got {value!r}"
        )
    return value


def fusion_mode_from_config(config: Mapping[str, Any]) -> str:
    """Node input fusion: historical concat or projected gated modalities."""
    fusion = ((config.get("ablation") or {}).get("fusion")) or {}
    value = str(fusion.get("mode") or "concat").lower()
    if value not in {"concat", "projected_gated"}:
        raise ValueError(
            f"ablation.fusion.mode must be 'concat' or 'projected_gated', got {value!r}"
        )
    return value


def node_loss_from_config(config: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Return node-loss reduction and modality weights."""
    fusion = ((config.get("ablation") or {}).get("fusion")) or {}
    mode = str(fusion.get("node_loss") or "global").lower()
    if mode not in {"global", "block_balanced"}:
        raise ValueError(
            "ablation.fusion.node_loss must be 'global' or 'block_balanced', "
            f"got {mode!r}"
        )
    raw = fusion.get("node_block_weights") or {}
    weights = {
        name: float(raw.get(name, 1.0)) for name in ("tfidf", "sbert", "extras")
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("Node block weights must be non-negative.")
    return mode, weights


def graph_identity_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the representation knobs that require a distinct graph bundle."""
    ablation = config["ablation"]
    embeddings = ablation.get("embeddings") or {}
    graph = ablation.get("graph") or {}
    llm_enabled = bool(ablation.get("llm_enrichment_enabled", True))
    size = str(ablation.get("enrichment_model_size", "large"))
    return {
        "llm_enrichment_enabled": llm_enabled,
        "enrichment_model_size": size if llm_enabled else None,
        "tfidf_enabled": bool(embeddings.get("tfidf_enabled", True)),
        "sbert_enabled": bool(embeddings.get("sbert_enabled", True)),
        "use_edge_features": bool(graph.get("use_edge_features", True)),
        "feature_contract": feature_contract_from_config(config),
        "unique_sequences": unique_sequences_from_config(config),
        "fit_on": fit_on_from_config(config),
        "split_protocol": split_protocol_from_config(config),
        "sbert_text": sbert_text_from_config(config),
    }


def graph_identity_from_meta(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract a graph identity from dataset_meta / bundle keys."""
    if not meta:
        return None
    embeddings = meta.get("embedding_flags") or meta.get("embeddings") or {}
    if "tfidf_enabled" in meta and "sbert_enabled" in meta:
        embeddings = {
            "tfidf_enabled": meta["tfidf_enabled"],
            "sbert_enabled": meta["sbert_enabled"],
        }
    llm_enabled = meta.get("llm_enrichment_enabled")
    if llm_enabled is None and "graph_identity" in meta:
        return _complete_graph_identity(meta["graph_identity"])
    if llm_enabled is None and not embeddings and "use_edge_features" not in meta:
        return None
    if llm_enabled is None:
        return None
    llm_enabled = bool(llm_enabled)
    size = meta.get("enrichment_model_size")
    return _complete_graph_identity(
        {
            "llm_enrichment_enabled": llm_enabled,
            "enrichment_model_size": (str(size) if size is not None else "large") if llm_enabled else None,
            "tfidf_enabled": bool(embeddings.get("tfidf_enabled", True)),
            "sbert_enabled": bool(embeddings.get("sbert_enabled", True)),
            "use_edge_features": bool(meta.get("use_edge_features", True)),
            "feature_contract": str(meta.get("feature_contract", "notebook_raw_v1")),
            "unique_sequences": _unique_sequences_from_meta(meta),
            "fit_on": str(_identity_field_from_meta(meta, "fit_on", "all")),
            "split_protocol": str(_identity_field_from_meta(meta, "split_protocol", "stratified")),
            "sbert_text": str(
                embeddings.get("sbert_text")
                or _identity_field_from_meta(meta, "sbert_text", "embedding_text")
            ),
        }
    )


def _unique_sequences_from_meta(meta: Mapping[str, Any]) -> bool:
    if "unique_sequences" in meta:
        return bool(meta["unique_sequences"])
    nested = meta.get("graph_identity")
    if isinstance(nested, Mapping) and "unique_sequences" in nested:
        return bool(nested["unique_sequences"])
    return False


def _identity_field_from_meta(meta: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in meta and meta[key] is not None:
        return meta[key]
    nested = meta.get("graph_identity")
    if isinstance(nested, Mapping) and nested.get(key) is not None:
        return nested[key]
    return default


def _complete_graph_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill identity keys; missing protocol fields are the transductive baseline."""
    identity = {key: raw.get(key) for key in GRAPH_IDENTITY_KEYS}
    identity["unique_sequences"] = bool(raw.get("unique_sequences", False))
    identity["fit_on"] = str(raw.get("fit_on") or "all")
    identity["split_protocol"] = str(raw.get("split_protocol") or "stratified")
    identity["sbert_text"] = str(raw.get("sbert_text") or "embedding_text")
    return identity


def identities_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare two graph identities with JSON-stable equality."""
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def assert_train_only_compatible(
    config: Mapping[str, Any],
    *,
    bundle_meta: Mapping[str, Any] | None,
    allow_legacy_baseline: bool = True,
) -> dict[str, Any]:
    """Refuse train-only when the config would need a different graph.

    Bundles without metadata are treated as the historical notebook baseline
    (hybrid TF-IDF+SBERT, LLM large, edge features, ``notebook_raw_v1``).
    """
    requested = graph_identity_from_config(config)
    actual = graph_identity_from_meta(bundle_meta)
    if actual is None:
        if allow_legacy_baseline and identities_match(requested, LEGACY_GRAPH_IDENTITY):
            return requested
        raise GraphIdentityError(
            "train-only requires dataset_meta.json (or bundle metadata) that matches "
            f"this config's graph identity {requested}. Rebuild the graph locally "
            "(Family A) or use a Family B config on the baseline bundle."
        )
    if not identities_match(requested, actual):
        raise GraphIdentityError(
            "train-only graph identity mismatch. "
            f"config={requested} bundle={actual}. "
            "Representation ablations must train the matching rebuilt graph."
        )
    return requested


def split_lock_id(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> str:
    """Stable short hash of the three index arrays."""
    payload = np.concatenate(
        [np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)]
    )
    return hashlib.sha256(payload.astype(np.int64).tobytes()).hexdigest()[:16]


def save_split_lock(
    path: str | Path,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    sequence_ids: list[str] | None = None,
    seed: int | None = None,
    protocol: str | None = None,
) -> str:
    """Persist a frozen 70/15/15 split and return its id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_id = split_lock_id(idx_train, idx_val, idx_test)
    payload: dict[str, Any] = {
        "idx_train": np.asarray(idx_train, dtype=np.int64),
        "idx_val": np.asarray(idx_val, dtype=np.int64),
        "idx_test": np.asarray(idx_test, dtype=np.int64),
        "n_total": np.int64(len(idx_train) + len(idx_val) + len(idx_test)),
        "split_lock_id": np.asarray(lock_id),
    }
    if seed is not None:
        payload["seed"] = np.int64(seed)
    if protocol is not None:
        payload["protocol"] = np.asarray(protocol)
    if sequence_ids is not None:
        payload["sequence_ids"] = np.asarray(sequence_ids)
    np.savez_compressed(path, **payload)
    return lock_id


def load_split_lock(path: str | Path) -> dict[str, Any]:
    """Load a split lock written by :func:`save_split_lock`."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as payload:
        lock = {
            "idx_train": np.asarray(payload["idx_train"], dtype=np.int64),
            "idx_val": np.asarray(payload["idx_val"], dtype=np.int64),
            "idx_test": np.asarray(payload["idx_test"], dtype=np.int64),
            "n_total": int(payload["n_total"]) if "n_total" in payload.files else None,
            "split_lock_id": str(payload["split_lock_id"]) if "split_lock_id" in payload.files else None,
        }
        if "sequence_ids" in payload.files:
            lock["sequence_ids"] = [str(item) for item in payload["sequence_ids"].tolist()]
        if "seed" in payload.files:
            lock["seed"] = int(payload["seed"])
        if "protocol" in payload.files:
            lock["protocol"] = str(payload["protocol"])
    if lock["split_lock_id"] is None:
        lock["split_lock_id"] = split_lock_id(
            lock["idx_train"], lock["idx_val"], lock["idx_test"]
        )
    if lock["n_total"] is None:
        lock["n_total"] = (
            len(lock["idx_train"]) + len(lock["idx_val"]) + len(lock["idx_test"])
        )
    return lock


def apply_split_lock(
    data_list: list,
    lock: Mapping[str, Any],
    *,
    sequence_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return locked indices after verifying they cover *data_list*."""
    n_total = len(data_list)
    expected = int(lock["n_total"])
    if n_total != expected:
        raise ValueError(
            f"Split lock n_total={expected} does not match built graphs ({n_total})."
        )
    if sequence_ids is not None and lock.get("sequence_ids"):
        if list(lock["sequence_ids"]) != list(sequence_ids):
            raise ValueError("Split lock sequence ids do not match the rebuilt graphs.")
    for name in ("idx_train", "idx_val", "idx_test"):
        indices = np.asarray(lock[name], dtype=np.int64)
        if indices.min(initial=0) < 0 or indices.max(initial=0) >= n_total:
            raise ValueError(f"Split lock {name} is out of range for {n_total} graphs.")
    return (
        np.asarray(lock["idx_train"], dtype=np.int64),
        np.asarray(lock["idx_val"], dtype=np.int64),
        np.asarray(lock["idx_test"], dtype=np.int64),
    )


def sequence_ids_from_graphs(data_list: list) -> list[str]:
    """Best-effort stable ids from PyG graphs (block_id / window_id)."""
    ids: list[str] = []
    for index, graph in enumerate(data_list):
        value = None
        for attribute in ("block_id", "window_id", "sequence_id", "graph_id"):
            if hasattr(graph, attribute):
                raw = getattr(graph, attribute)
                value = raw.item() if hasattr(raw, "item") else raw
                break
        ids.append(str(index if value is None else value))
    return ids


def load_bundle_meta(graph_path: str | Path) -> dict[str, Any] | None:
    """Load dataset_meta from a sidecar JSON or keys stored in the ``.pt`` bundle."""
    graph_path = Path(graph_path)
    sidecar = graph_path.with_name("dataset_meta.json")
    if not sidecar.exists() and graph_path.suffix == ".gz":
        sidecar = graph_path.with_suffix("").with_name("dataset_meta.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    parent_meta = graph_path.parent / "dataset_meta.json"
    if parent_meta.exists():
        return json.loads(parent_meta.read_text())
    try:
        import torch

        bundle = torch.load(graph_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(bundle, dict):
        return None
    meta = bundle.get("dataset_meta")
    if isinstance(meta, dict):
        return meta
    extracted = {
        key: bundle[key]
        for key in (
            "node_dim",
            "edge_dim",
            "embed_dim",
            "feature_contract",
            "llm_enrichment_enabled",
            "enrichment_model_size",
            "use_edge_features",
            "embedding_flags",
            "graph_identity",
            "split_lock_id",
        )
        if key in bundle
    }
    return extracted or None


def gzip_file(source: str | Path, destination: str | Path) -> Path:
    """Compress *source* to *destination* (``.pt.gz``)."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, gzip.open(destination, "wb", compresslevel=6) as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=16 * 1024 * 1024)
    return destination


def gunzip_file(source: str | Path, destination: str | Path | None = None) -> Path:
    """Decompress a ``.gz`` graph bundle to *destination* (default: strip ``.gz``)."""
    source = Path(source)
    if destination is None:
        if source.suffix != ".gz":
            return source
        destination = source.with_suffix("")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=16 * 1024 * 1024)
    return destination


def sha256_file(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return the hex digest of *path* without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: str | Path) -> dict[str, Any]:
    """Size + sha256 for a campaign manifest entry."""
    path = Path(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Load a Family A/B matrix YAML."""
    matrix_path = Path(path)
    matrix = yaml.safe_load(matrix_path.read_text()) or {}
    if not isinstance(matrix, dict):
        raise ValueError(f"Matrix must be a mapping: {matrix_path}")
    return matrix


def enabled_experiments(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return enabled experiment records from a matrix mapping."""
    experiments = matrix.get("experiments") or []
    return [item for item in experiments if item.get("enabled", True)]


def experiment_seed_pairs(
    matrix: Mapping[str, Any], default_seed: int
) -> list[tuple[dict[str, Any], int]]:
    """Cartesian product of enabled arms and declared training seeds."""
    raw_seeds = matrix.get("seeds") or [default_seed]
    seeds = [int(seed) for seed in raw_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Matrix seeds must be unique.")
    return [
        (experiment, seed)
        for experiment in enabled_experiments(matrix)
        for seed in seeds
    ]


def experiment_requires_graph_rebuild(
    experiment: Mapping[str, Any],
    matrix: Mapping[str, Any] | None = None,
) -> bool:
    """True when this arm cannot share a frozen train-only graph."""
    if "requires_graph_rebuild" in experiment:
        return bool(experiment["requires_graph_rebuild"])
    if matrix and "requires_graph_rebuild" in matrix:
        return bool(matrix["requires_graph_rebuild"])
    return bool(matrix and matrix.get("family") == "representation")


def _component_aucs(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    has_both = len(np.unique(labels)) == 2
    return {
        "pr_auc": float(average_precision_score(labels, scores)) if has_both else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if has_both else 0.0,
    }


def component_metrics(
    labels: np.ndarray,
    *,
    combined: np.ndarray,
    structure: np.ndarray,
    node: np.ndarray,
    edge: np.ndarray,
    predictions: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    node_blocks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """PR/ROC for each reconstruction head plus TP dominance shares."""
    metrics: dict[str, Any] = {
        "combined": _component_aucs(labels, combined),
        "structure": _component_aucs(labels, structure),
        "node": _component_aucs(labels, node),
        "edge": _component_aucs(labels, edge),
    }
    for name, values in (node_blocks or {}).items():
        array = np.asarray(values)
        metrics[name] = {
            **_component_aucs(labels, array),
            "mean_normal": float(array[labels == 0].mean()) if (labels == 0).any() else 0.0,
            "mean_anomaly": float(array[labels == 1].mean()) if (labels == 1).any() else 0.0,
        }
    weighted = np.column_stack(
        [alpha * structure, beta * node, gamma * edge]
    )
    totals = weighted.sum(axis=1, keepdims=True)
    shares = np.divide(
        weighted,
        np.maximum(totals, 1e-12),
        out=np.zeros_like(weighted, dtype=float),
        where=totals > 0,
    )
    true_positives = (labels == 1) & (predictions == 1)
    if true_positives.any():
        mean_share = shares[true_positives].mean(axis=0)
        dominant = np.argmax(shares[true_positives], axis=1)
        names = ("structure", "node", "edge")
        metrics["true_positive_share"] = {
            name: float(mean_share[index]) for index, name in enumerate(names)
        }
        metrics["true_positive_dominance_count"] = {
            name: int((dominant == index).sum()) for index, name in enumerate(names)
        }
        metrics["n_true_positives"] = int(true_positives.sum())
    else:
        metrics["true_positive_share"] = {
            "structure": 0.0, "node": 0.0, "edge": 0.0
        }
        metrics["true_positive_dominance_count"] = {
            "structure": 0, "node": 0, "edge": 0
        }
        metrics["n_true_positives"] = 0
    return metrics


def write_eval_pack(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    history_epochs: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    structure: np.ndarray,
    node: np.ndarray,
    edge: np.ndarray,
    threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
    node_blocks: Mapping[str, np.ndarray] | None = None,
    graph_ids: list[str] | None = None,
    config: Mapping[str, Any] | None = None,
    campaign_meta: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the per-run JSON/CSV/figure pack under *output_dir*."""
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    scores_dir = output_dir / "scores"
    figures.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    predictions = (np.asarray(scores) > threshold).astype(int)
    labels = np.asarray(labels)
    structure = np.asarray(structure)
    node = np.asarray(node)
    edge = np.asarray(edge)
    components = component_metrics(
        labels,
        combined=np.asarray(scores),
        structure=structure,
        node=node,
        edge=edge,
        predictions=predictions,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        node_blocks=node_blocks,
    )
    component_path = output_dir / "component_metrics.json"
    component_path.write_text(json.dumps(components, indent=2))

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps({"epochs": history_epochs}, indent=2))

    if config is not None:
        config_path = output_dir / "config.yaml"
        serialisable = {key: value for key, value in config.items() if key != "__pipeline__"}
        config_path.write_text(yaml.safe_dump(serialisable, sort_keys=False))
    else:
        config_path = output_dir / "config.yaml"

    score_columns: dict[str, Any] = {
            "graph_id": graph_ids if graph_ids is not None else list(range(len(labels))),
            "label": labels.astype(int),
            "prediction": predictions,
            "score": np.asarray(scores),
            "structure": structure,
            "node": node,
            "edge": edge,
            "weighted_structure": alpha * structure,
            "weighted_node": beta * node,
            "weighted_edge": gamma * edge,
    }
    for name, values in (node_blocks or {}).items():
        score_columns[name] = np.asarray(values)
    score_frame = pd.DataFrame(score_columns)
    scores_csv = scores_dir / "test_component_scores.csv"
    score_frame.to_csv(scores_csv, index=False)

    manifest_path = output_dir / "manifest.json"
    if campaign_meta is not None:
        manifest_path.write_text(json.dumps(dict(campaign_meta), indent=2, default=str))
    else:
        manifest_path.write_text("{}")

    written: dict[str, Path] = {
        "component_metrics": component_path,
        "history": history_path,
        "config": config_path,
        "scores_csv": scores_csv,
        "manifest": manifest_path,
        "metrics": output_dir / "metrics.json",
    }
    (output_dir / "metrics.json").write_text(json.dumps(dict(metrics), indent=2, default=str))
    written.update(
        _write_eval_figures(
            figures,
            history_epochs=history_epochs,
            labels=labels,
            scores=np.asarray(scores),
            predictions=predictions,
            threshold=threshold,
            structure=structure,
            node=node,
            edge=edge,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
    )
    return written


def _write_eval_figures(
    figures: Path,
    *,
    history_epochs: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    structure: np.ndarray,
    node: np.ndarray,
    edge: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[str, Path]:
    """Save the standard per-run PNG figures. Best-effort if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            average_precision_score,
            confusion_matrix,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
    except ImportError:
        return {}

    written: dict[str, Path] = {}
    if history_epochs:
        epochs = [int(entry["epoch"]) for entry in history_epochs]
        figure, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        axes[0].plot(epochs, [entry["train_total_loss"] for entry in history_epochs], color="black", linewidth=2)
        axes[0].set(title="Total training loss", xlabel="Epoch", ylabel="Loss")
        for key, color, label in (
            ("train_structure_loss", "#4C72B0", "structure"),
            ("train_node_loss", "#55A868", "node"),
            ("train_edge_loss", "#C44E52", "edge"),
        ):
            axes[1].plot(epochs, [entry[key] for entry in history_epochs], linewidth=2, color=color, label=label)
        axes[1].set(title="Reconstruction components", xlabel="Epoch", ylabel="Loss")
        axes[1].legend()
        if "val_f1" in history_epochs[0]:
            axes[2].plot(epochs, [entry["val_f1"] for entry in history_epochs], linewidth=2, color="#8172B2", label="Val F1")
            axes[2].plot(epochs, [entry["val_pr_auc"] for entry in history_epochs], linewidth=1.5, color="#DD8452", label="Val PR-AUC")
            axes[2].plot(epochs, [entry["val_roc_auc"] for entry in history_epochs], linewidth=1.5, color="#937860", label="Val ROC-AUC")
            axes[2].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1.05))
            axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.tight_layout()
        path = figures / "training_history.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written["training_history"] = path

    figure, axes = plt.subplots(1, 1, figsize=(7, 4.5))
    for class_id, color, title in ((0, "#4C72B0", "Normal"), (1, "#C44E52", "Anomaly")):
        subset = scores[labels == class_id]
        if len(subset):
            axes.hist(subset, density=True, bins=40, alpha=0.45, color=color, label=f"{title} (n={len(subset)})")
    axes.axvline(threshold, color="black", linestyle="--", label="validation threshold")
    axes.set(title="Held-out anomaly-score distribution", xlabel="Weighted reconstruction error")
    axes.legend()
    axes.grid(alpha=0.25)
    figure.tight_layout()
    dist_path = figures / "test_score_distribution.png"
    figure.savefig(dist_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    written["score_distribution"] = dist_path

    figure, axis = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions, labels=[0, 1]),
        display_labels=["Normal", "Anomaly"],
    ).plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_title("Held-out confusion matrix")
    figure.tight_layout()
    cm_path = figures / "confusion_matrix.png"
    figure.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    written["confusion_matrix"] = cm_path

    if len(np.unique(labels)) == 2:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(recall, precision, linewidth=2, label=f"PR-AUC = {average_precision_score(labels, scores):.4f}")
        axes[0].axhline(float(labels.mean()), color="gray", linestyle="--", label="positive-class rate")
        axes[0].set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
        axes[0].legend()
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        axes[1].plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"ROC-AUC = {roc_auc_score(labels, scores):.4f}")
        axes[1].plot([0, 1], [0, 1], "--", color="gray", label="random")
        axes[1].set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        axes[1].legend()
        figure.tight_layout()
        pr_path = figures / "test_pr_roc.png"
        figure.savefig(pr_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written["pr_roc"] = pr_path

    weighted = pd.DataFrame(
        {
            "Structure": alpha * structure,
            "Node": beta * node,
            "Edge": gamma * edge,
            "label": labels,
            "prediction": predictions,
        }
    )
    true_positives = weighted[(weighted.label == 1) & (weighted.prediction == 1)]
    if len(true_positives):
        contribution = true_positives[["Structure", "Node", "Edge"]].div(
            true_positives[["Structure", "Node", "Edge"]].sum(axis=1), axis=0
        ).fillna(0)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        average_contribution = contribution.mean().mul(100)
        axes[0].pie(average_contribution, labels=list(average_contribution.index), autopct="%1.1f%%")
        axes[0].set_title(f"Average weighted contribution — true positives (n={len(true_positives)})")
        dominant = contribution.idxmax(axis=1).value_counts().reindex(["Structure", "Node", "Edge"], fill_value=0)
        axes[1].bar(dominant.index, dominant.values, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[1].set(title="Dominant component per true positive", ylabel="Graphs")
        axes[1].grid(axis="y", alpha=0.25)
        figure.tight_layout()
        contrib_path = figures / "component_contribution.png"
        figure.savefig(contrib_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written["component_contribution"] = contrib_path
    return written


def write_campaign_report(
    workspace: str | Path,
    *,
    dataset: str,
    campaign_id: str,
    baseline_name: str = "baseline_full",
) -> Path:
    """Build leaderboard + comparison plots from completed run directories."""
    workspace = Path(workspace)
    output_root = workspace / "outputs" / dataset
    campaign_dir = output_root / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    prefix = f"{campaign_id}_"
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.name in {"campaigns", "notebook_reports"} or run_dir.name.endswith("_ablation_matrix"):
            continue
        if not run_dir.name.startswith(prefix) and run_dir.name != campaign_id:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("campaign_id") != campaign_id:
                continue
        metrics = json.loads(metrics_path.read_text())
        manifest = {}
        if (run_dir / "manifest.json").exists():
            manifest = json.loads((run_dir / "manifest.json").read_text())
        components = {}
        if (run_dir / "component_metrics.json").exists():
            components = json.loads((run_dir / "component_metrics.json").read_text())
        history = {}
        if (run_dir / "history.json").exists():
            history = json.loads((run_dir / "history.json").read_text())
        name = manifest.get("experiment_name") or run_dir.name.removeprefix(prefix)
        records.append(
            {
                "name": name,
                "run_id": run_dir.name,
                "status": "OK",
                "dataset": dataset,
                "campaign_id": campaign_id,
                "family": manifest.get("family"),
                "seed": manifest.get("seed", metrics.get("seed")),
                "split_protocol": manifest.get("split_protocol"),
                "oov_graph_rate": manifest.get("oov_graph_rate"),
                "oov_line_rate": manifest.get("oov_line_rate"),
                "run_dir": str(run_dir),
                "test_f1": metrics.get("test_f1"),
                "test_pr_auc": metrics.get("test_pr_auc"),
                "test_roc_auc": metrics.get("test_roc_auc"),
                "val_f1": metrics.get("val_f1"),
                "val_pr_auc": metrics.get("val_pr_auc"),
                "val_roc_auc": metrics.get("val_roc_auc"),
                "test_precision": metrics.get("test_precision"),
                "test_recall": metrics.get("test_recall"),
                "best_threshold": metrics.get("best_threshold"),
                "component_metrics": components,
                "history": _history_from_epochs(history.get("epochs") or metrics.get("history")),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No completed runs found for campaign {campaign_id!r} under {output_root}"
        )

    frame = pd.DataFrame(records)
    for column in ("test_f1", "test_pr_auc", "test_roc_auc"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("test_pr_auc", ascending=False, na_position="last")
    csv_path = campaign_dir / "leaderboard.csv"
    json_path = campaign_dir / "leaderboard.json"
    frame.drop(columns=["component_metrics", "history"], errors="ignore").to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2, default=str))

    summary_frame, paired_frame = _seeded_campaign_statistics(frame, baseline_name)
    summary_frame.to_csv(campaign_dir / "summary_by_arm.csv", index=False)
    paired_frame.to_csv(campaign_dir / "paired_bootstrap_ci.csv", index=False)

    baseline_rows = frame[frame["name"] == baseline_name]
    delta_path = campaign_dir / "delta_vs_baseline.csv"
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        delta_rows = []
        for _, row in frame.iterrows():
            delta_rows.append(
                {
                    "name": row["name"],
                    "test_f1_delta": _delta(row.get("test_f1"), baseline.get("test_f1")),
                    "test_pr_auc_delta": _delta(row.get("test_pr_auc"), baseline.get("test_pr_auc")),
                    "test_roc_auc_delta": _delta(row.get("test_roc_auc"), baseline.get("test_roc_auc")),
                    "test_f1_rel": _relative(row.get("test_f1"), baseline.get("test_f1")),
                    "test_pr_auc_rel": _relative(row.get("test_pr_auc"), baseline.get("test_pr_auc")),
                    "test_roc_auc_rel": _relative(row.get("test_roc_auc"), baseline.get("test_roc_auc")),
                }
            )
        pd.DataFrame(delta_rows).to_csv(delta_path, index=False)

    _write_campaign_figures(campaign_dir, records, baseline_name=baseline_name)
    readme = campaign_dir / "README.md"
    readme.write_text(_campaign_readme(campaign_id, dataset, frame, baseline_name))
    return json_path


def _seeded_campaign_statistics(
    frame: pd.DataFrame,
    baseline_name: str,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean/std by arm and paired seed-bootstrap deltas vs baseline."""
    metrics = ("test_f1", "test_pr_auc", "test_roc_auc")
    summary_rows: list[dict[str, Any]] = []
    for name, group in frame.groupby("name", sort=True):
        row: dict[str, Any] = {"name": name, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    baseline = frame[frame["name"] == baseline_name]
    if not baseline.empty and baseline["seed"].notna().all():
        rng = np.random.default_rng(bootstrap_seed)
        for name, group in frame.groupby("name", sort=True):
            merged = baseline[["seed", *metrics]].merge(
                group[["seed", *metrics]], on="seed", suffixes=("_baseline", "_arm")
            )
            for metric in metrics:
                delta = (
                    pd.to_numeric(merged[f"{metric}_arm"], errors="coerce")
                    - pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
                ).dropna().to_numpy(dtype=float)
                if not len(delta):
                    continue
                sampled = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
                paired_rows.append(
                    {
                        "name": name,
                        "metric": metric,
                        "n_paired_seeds": int(len(delta)),
                        "mean_delta": float(delta.mean()),
                        "ci95_low": float(np.quantile(sampled, 0.025)),
                        "ci95_high": float(np.quantile(sampled, 0.975)),
                        "bootstrap_unit": "training_seed",
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def _history_from_epochs(history: Any) -> dict[str, list[float]]:
    if isinstance(history, dict) and "total" in history:
        return history
    if isinstance(history, list):
        return {
            "total": [float(entry.get("train_total_loss", 0.0)) for entry in history],
            "structure": [float(entry.get("train_structure_loss", 0.0)) for entry in history],
            "node": [float(entry.get("train_node_loss", 0.0)) for entry in history],
            "edge": [float(entry.get("train_edge_loss", 0.0)) for entry in history],
        }
    return {}


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline):
        return None
    return round(float(value) - float(baseline), 6)


def _relative(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline) or float(baseline) == 0:
        return None
    return round((float(value) - float(baseline)) / abs(float(baseline)), 6)


def _write_campaign_figures(
    campaign_dir: Path,
    records: list[dict[str, Any]],
    *,
    baseline_name: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return

    ok = [record for record in records if record.get("status") == "OK"]
    if not ok:
        return
    names = [record["name"] for record in ok]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, label in zip(
        axes,
        ("test_f1", "test_pr_auc", "test_roc_auc"),
        ("Test F1", "Test PR-AUC", "Test ROC-AUC"),
    ):
        values = [float(record.get(metric) or 0.0) for record in ok]
        bars = axis.barh(x, values, color=plt.cm.tab10.colors[: len(names)], edgecolor="white", height=0.6)
        if values:
            bars[int(np.argmax(values))].set_edgecolor("#111")
            bars[int(np.argmax(values))].set_linewidth(2)
        for bar, value in zip(bars, values):
            axis.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8)
        axis.set_yticks(x)
        axis.set_yticklabels(names, fontsize=9)
        axis.set_xlabel(label)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.3)
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    figure.suptitle("Ablation comparison", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(campaign_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    histories = {record["name"]: record.get("history") or {} for record in ok if record.get("history")}
    if histories:
        n_exp = len(histories)
        figure, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4), squeeze=False)
        for axis, (name, hist) in zip(axes[0], histories.items()):
            epochs = range(1, len(hist.get("total") or []) + 1)
            axis.plot(epochs, hist.get("total") or [], "ko-", lw=2, ms=4, label="Total")
            axis.plot(epochs, hist.get("structure") or [], "bo--", lw=1.4, ms=3, label="Structure")
            axis.plot(epochs, hist.get("node") or [], "go-.", lw=1.4, ms=3, label="Node")
            axis.plot(epochs, hist.get("edge") or [], "ro:", lw=1.4, ms=3, label="Edge")
            axis.set_title(name, fontsize=9)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.legend(fontsize=7)
            axis.grid(alpha=0.3)
        figure.suptitle("Training loss curves", fontsize=13, fontweight="bold")
        figure.tight_layout()
        figure.savefig(campaign_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

    component_rows = []
    for record in ok:
        combined = (record.get("component_metrics") or {}).get("combined") or {}
        node = (record.get("component_metrics") or {}).get("node") or {}
        edge = (record.get("component_metrics") or {}).get("edge") or {}
        structure = (record.get("component_metrics") or {}).get("structure") or {}
        if combined or node:
            component_rows.append(
                {
                    "name": record["name"],
                    "combined": combined.get("roc_auc", 0.0),
                    "structure": structure.get("roc_auc", 0.0),
                    "node": node.get("roc_auc", 0.0),
                    "edge": edge.get("roc_auc", 0.0),
                }
            )
    if component_rows:
        figure, axis = plt.subplots(figsize=(10, 5))
        index = np.arange(len(component_rows))
        width = 0.2
        for offset, key, color in (
            (-1.5, "combined", "black"),
            (-0.5, "structure", "#4C72B0"),
            (0.5, "node", "#55A868"),
            (1.5, "edge", "#C44E52"),
        ):
            axis.bar(index + offset * width, [row[key] for row in component_rows], width, label=key, color=color)
        axis.set_xticks(index)
        axis.set_xticklabels([row["name"] for row in component_rows], rotation=30, ha="right")
        axis.set_ylabel("ROC-AUC")
        axis.set_ylim(0, 1.05)
        axis.legend()
        axis.set_title("Component ROC-AUC vs combined")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(campaign_dir / "component_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def _campaign_readme(
    campaign_id: str,
    dataset: str,
    frame: pd.DataFrame,
    baseline_name: str,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {dataset.upper()} ablation campaign `{campaign_id}`",
        "",
        f"Generated {generated}. Primary ranking metric: **test PR-AUC**.",
        "",
        "| name | test_f1 | test_pr_auc | test_roc_auc | val_f1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        marker = " **(baseline)**" if row["name"] == baseline_name else ""
        lines.append(
            f"| {row['name']}{marker} | {row.get('test_f1'):.4f} | {row.get('test_pr_auc'):.4f} "
            f"| {row.get('test_roc_auc'):.4f} | {row.get('val_f1'):.4f} |"
            if pd.notna(row.get("test_f1"))
            else f"| {row['name']}{marker} |  |  |  |  |"
        )
    lines.extend(
        [
            "",
            "Artifacts per run live under `outputs/{dataset}/{run_id}/` "
            "(metrics, figures, scores, checkpoint). This folder is the campaign rollup.",
            "",
        ]
    )
    return "\n".join(lines)


def write_campaign_manifest(
    campaign_dir: str | Path,
    *,
    campaign_id: str,
    dataset: str,
    family: str,
    graphs: list[dict[str, Any]],
    split_lock: str | Path | None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` for a locally prepared campaign directory."""
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_id": campaign_id,
        "dataset": dataset,
        "family": family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_lock": str(split_lock) if split_lock else None,
        "graphs": graphs,
    }
    if extra:
        payload.update(dict(extra))
    path = campaign_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
