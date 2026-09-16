"""Stage 5 — PyTorch Geometric dataset preparation.

Converts per-sequence DataFrames and pre-computed template embeddings into
:class:`torch_geometric.data.Data` objects ready for GNN training.

Embedding strategy is controlled by two boolean ablation flags:
    ``tfidf_enabled``  — include TF-IDF vectors (structural/token features)
    ``sbert_enabled``  — include Sentence-BERT vectors (semantic features)
At least one must be True.

Public API
----------
    compute_embeddings(templates_data, cluster_to_enriched, *, ...) → (ndarray, cids, vec)
    build_pyg_dataset(sequences, block_labels, cluster_embeddings, *, ...) → list[Data]
    split_dataset(all_data, seed, *, ...) → (idx_train, idx_val, idx_test)
    save_graph_dataset(all_data, ..., path, *, ...)
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
HdfsFeatureContract = Literal["notebook_raw_v1", "stabilized_v2"]


class MissingClusterEmbedding(ValueError):
    """Raised when a sequence references a cluster ID absent from frozen embeddings."""

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id = cluster_id
        super().__init__(f"Cluster {cluster_id} has no frozen embedding.")


# ── Embedding computation ─────────────────────────────────────────────────────


def compute_embeddings(
    templates_data: list[dict],
    cluster_to_enriched: dict,
    *,
    tfidf_enabled: bool = True,
    sbert_enabled: bool = True,
) -> tuple[np.ndarray, list[int], Any]:
    """Compute hybrid (TF-IDF + SBERT) template embedding matrix.

    Parameters
    ----------
    templates_data : list[dict]
        Template metadata list (from parser/enrichment stage).
    cluster_to_enriched : dict
        Mapping ``cluster_id → enrichment dict``.
    tfidf_enabled, sbert_enabled : bool
        Ablation toggles.  At least one must be True.

    Returns
    -------
    hybrid_embeddings : np.ndarray, shape (n_templates, embed_dim)
    all_cids : list[int]
        Ordered cluster IDs matching embedding rows.
    tfidf_vectorizer : TfidfVectorizer | None
        Fitted vectorizer (serialisable for reuse); None if TF-IDF disabled.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not tfidf_enabled and not sbert_enabled:
        raise ValueError("At least one of tfidf_enabled or sbert_enabled must be True.")

    all_cids: list[int] = [t["cluster_id"] for t in templates_data]
    all_templates: list[str] = [t["template"] for t in templates_data]
    cluster_to_template: dict[int, str] = dict(zip(all_cids, all_templates))

    parts: list[np.ndarray] = []
    tfidf_vectorizer: Any = None

    # ── TF-IDF (structural token features) ───────────────────────────────────
    if tfidf_enabled:
        tfidf_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s]+")
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_templates)
        tfidf_dense = tfidf_matrix.toarray().astype(np.float32)
        parts.append(tfidf_dense)
        logger.info(
            "TF-IDF: %d-dim vectors for %d templates", tfidf_dense.shape[1], len(all_cids)
        )

    # ── Sentence-BERT (semantic features on enriched text) ───────────────────
    if sbert_enabled:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        enriched_texts: list[str] = []
        for cid in all_cids:
            info = cluster_to_enriched.get(cid)
            if info:
                text = info.get("embedding_text", "").strip()
            else:
                text = cluster_to_template.get(cid, "").strip()
            if not text:
                text = cluster_to_template.get(cid, "unknown log template")
            enriched_texts.append(text)

        sbert_emb = sbert_model.encode(
            enriched_texts, show_progress_bar=True, normalize_embeddings=True
        )
        sbert_emb = np.nan_to_num(
            sbert_emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        parts.append(sbert_emb)
        logger.info(
            "SBERT: %d-dim vectors for %d templates", sbert_emb.shape[1], len(all_cids)
        )

    hybrid_embeddings = np.hstack(parts) if len(parts) > 1 else parts[0]
    logger.info("Hybrid embedding dim: %d", hybrid_embeddings.shape[1])
    return hybrid_embeddings, all_cids, tfidf_vectorizer


# ── PyG graph construction ────────────────────────────────────────────────────


def build_pyg_dataset(
    sequences: dict,
    block_labels: dict,
    cluster_embeddings: dict[int, np.ndarray],
    *,
    use_edge_features: bool = True,
    dataset: str = "bgl",
    missing_embedding: Literal["zero", "fail"] = "zero",
    on_graph_error: Literal["skip", "fail"] = "skip",
    hdfs_feature_contract: HdfsFeatureContract = "stabilized_v2",
) -> list:
    """Convert sequences to a list of :class:`torch_geometric.data.Data` objects.

    Parameters
    ----------
    sequences : dict
        ``{sequence_id: DataFrame}`` produced by the sequencer.
    block_labels : dict
        ``{sequence_id: 0|1}`` anomaly labels.
    cluster_embeddings : dict
        ``{cluster_id: np.ndarray}`` embedding vectors.
    use_edge_features : bool
        When False, edge_attr carries only the log-scaled transition count.
    dataset : str
        ``"bgl"`` (uses ``unix_ts`` + ``window_id``) or
        ``"hdfs"`` (uses ``timestamp`` + ``block_id``).
    hdfs_feature_contract : str
        HDFS feature encoding declared by the model package. ``"notebook_raw_v1"``
        preserves the approved Colab baseline; ``"stabilized_v2"`` uses the
        current log-scaled feature construction.

    Returns
    -------
    list[torch_geometric.data.Data]
    """
    embed_dim = next(iter(cluster_embeddings.values())).shape[0]
    node_extra_dim = 9
    node_dim = embed_dim + node_extra_dim
    edge_dim = 10 if use_edge_features else 1

    all_data = []
    skipped = 0
    t0 = time.time()
    normalized_embeddings = {int(cid): vector for cid, vector in cluster_embeddings.items()}
    if missing_embedding == "fail":
        for seq in sequences.values():
            for cluster_id in seq["cluster_id"].tolist():
                if int(cluster_id) not in normalized_embeddings:
                    raise MissingClusterEmbedding(int(cluster_id))

    for wid, seq in sequences.items():
        label = block_labels.get(wid, 0)
        try:
            data = _seq_to_pyg(
                seq,
                label,
                normalized_embeddings,
                embed_dim=embed_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                use_edge_features=use_edge_features,
                dataset=dataset,
                missing_embedding=missing_embedding,
                hdfs_feature_contract=hdfs_feature_contract,
            )
            all_data.append(data)
        except MissingClusterEmbedding:
            raise
        except Exception as exc:
            if on_graph_error == "fail":
                raise
            skipped += 1
            if skipped <= 5:
                logger.warning("Skipped sequence %s: %s", wid, exc)

    logger.info(
        "Built %d PyG graphs in %.1fs  (skipped %d)",
        len(all_data),
        time.time() - t0,
        skipped,
    )
    return all_data


def split_dataset(
    all_data: list,
    seed: int = 42,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / val / test split (default 70 / 15 / 15).

    Returns
    -------
    idx_train, idx_val, idx_test : np.ndarray
    """
    from sklearn.model_selection import train_test_split

    all_labels = np.array([d.y.item() for d in all_data])
    indices = np.arange(len(all_data))
    test_ratio = 1.0 - train_ratio - val_ratio

    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=all_labels,
    )
    labels_temp = all_labels[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=labels_temp,
    )
    return idx_train, idx_val, idx_test


def save_graph_dataset(
    all_data: list,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    path: str | Path,
    *,
    node_dim: int,
    edge_dim: int,
    embed_dim: int,
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    """Persist the graph dataset bundle to ``path`` via ``torch.save``."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "data_list": all_data,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "embed_dim": embed_dim,
    }
    if dataset_meta:
        payload["dataset_meta"] = dataset_meta
        for key in (
            "feature_contract",
            "llm_enrichment_enabled",
            "enrichment_model_size",
            "use_edge_features",
            "embedding_flags",
            "graph_identity",
            "split_lock_id",
        ):
            if key in dataset_meta:
                payload[key] = dataset_meta[key]
    torch.save(payload, path)
    logger.info("Graph dataset saved → %s  (%.1f MB)", path, path.stat().st_size / 1e6)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_numeric(x: object) -> bool:
    """Return True only for finite non-NaN/Inf numeric values."""
    try:
        v = float(cast(Any, x))
        return np.isfinite(v)
    except (ValueError, TypeError):
        return False


def _seq_to_pyg(
    seq: pd.DataFrame,
    label: int,
    cluster_embeddings: dict,
    *,
    embed_dim: int,
    node_dim: int,
    edge_dim: int,
    use_edge_features: bool,
    dataset: str,
    missing_embedding: Literal["zero", "fail"] = "zero",
    hdfs_feature_contract: HdfsFeatureContract = "stabilized_v2",
):
    """Convert a single sequence DataFrame to a PyG Data object."""
    import torch
    from torch_geometric.data import Data

    ts_col = "unix_ts" if dataset.lower() == "bgl" else "timestamp"
    id_col = "window_id" if dataset.lower() == "bgl" else "block_id"

    seq_id = seq[id_col].iloc[0]
    cids = seq["cluster_id"].tolist()
    params = seq["parameters"].tolist() if "parameters" in seq.columns else [[] for _ in cids]
    ts = seq[ts_col].tolist() if ts_col in seq.columns else [None] * len(cids)
    n = len(cids)

    # ── Node aggregation ──────────────────────────────────────────────────────
    node_params: dict = defaultdict(list)
    node_count = Counter(cids)
    node_positions: dict = defaultdict(list)

    for i, (cid, p) in enumerate(zip(cids, params)):
        node_params[cid].extend(p if isinstance(p, list) else [])
        node_positions[cid].append(i / max(n - 1, 1))

    unique_cids = [int(cid) for cid in dict.fromkeys(cids)]
    cid_to_idx = {cid: idx for idx, cid in enumerate(unique_cids)}
    num_nodes = len(unique_cids)

    node_feats = np.zeros((num_nodes, node_dim), dtype=np.float32)
    for idx, cid in enumerate(unique_cids):
        if cid not in cluster_embeddings:
            if missing_embedding == "fail":
                raise MissingClusterEmbedding(int(cid))
            emb = np.zeros(embed_dim, dtype=np.float32)
        else:
            emb = cluster_embeddings[cid]
        nums = [float(x) for x in node_params[cid] if _is_numeric(x)]
        pos = np.array(node_positions[cid])

        node_feats[idx, :embed_dim] = emb
        if dataset.lower() == "hdfs" and hdfs_feature_contract == "notebook_raw_v1":
            node_feats[idx, embed_dim + 0] = float(node_count[cid])
            node_feats[idx, embed_dim + 1] = float(len(node_params[cid]))
            node_feats[idx, embed_dim + 2] = float(np.mean(nums)) if nums else 0.0
            node_feats[idx, embed_dim + 3] = float(np.max(nums)) if nums else 0.0
        else:
            node_feats[idx, embed_dim + 0] = float(np.log1p(node_count[cid]))
            node_feats[idx, embed_dim + 1] = float(np.log1p(len(node_params[cid])))
            node_feats[idx, embed_dim + 2] = (
                float(np.log1p(abs(np.mean(nums)))) if nums else 0.0
            )
            node_feats[idx, embed_dim + 3] = float(np.log1p(abs(np.max(nums)))) if nums else 0.0
        node_feats[idx, embed_dim + 4] = float(pos.min())
        node_feats[idx, embed_dim + 5] = float(pos.max())
        node_feats[idx, embed_dim + 6] = float(pos.mean())
        node_feats[idx, embed_dim + 7] = float(pos.std()) if len(pos) > 1 else 0.0
        node_feats[idx, embed_dim + 8] = float(pos.max() - pos.min())

    node_feats = np.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Edge aggregation ──────────────────────────────────────────────────────
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
                if dataset.lower() == "hdfs" and hdfs_feature_contract == "notebook_raw_v1":
                    delta = (t_dst - t_src).total_seconds()
                else:
                    delta = float(t_dst) - float(t_src)
                edge_deltas[(src, dst)].append(delta)
            except (AttributeError, TypeError, ValueError):
                if (src, dst) not in edge_deltas:
                    edge_deltas[(src, dst)]
        elif (src, dst) not in edge_deltas:
            edge_deltas[(src, dst)]

    src_list, dst_list, edge_feats_list = [], [], []
    for (src, dst) in edge_src_pos:
        deltas = edge_deltas.get((src, dst), [])
        s_pos = np.array(edge_src_pos[(src, dst)])
        d_pos = np.array(edge_dst_pos[(src, dst)])

        if use_edge_features:
            ef = np.zeros(edge_dim, dtype=np.float32)
            if dataset.lower() == "hdfs" and hdfs_feature_contract == "notebook_raw_v1":
                ef[0] = float(len(s_pos))
                if deltas:
                    arr = np.array(deltas, dtype=np.float64)
                    ef[1] = float(arr.min())
                    ef[2] = float(np.percentile(arr, 25))
                    ef[3] = float(np.median(arr))
                    ef[4] = float(np.percentile(arr, 75))
                    ef[5] = float(arr.max())
                    ef[6] = float(arr.std())
                else:
                    ef[1:7] = [-1, -1, -1, -1, -1, 0]
            else:
                ef[0] = float(np.log1p(len(s_pos)))
                if deltas:
                    arr = np.clip(np.array(deltas, dtype=np.float64), 0.0, None)
                    ef[1] = float(np.log1p(arr.min()))
                    ef[2] = float(np.log1p(np.percentile(arr, 25)))
                    ef[3] = float(np.log1p(np.median(arr)))
                    ef[4] = float(np.log1p(np.percentile(arr, 75)))
                    ef[5] = float(np.log1p(arr.max()))
                    ef[6] = float(np.log1p(arr.std()))
            ef[7] = float(s_pos.mean())
            ef[8] = float(d_pos.mean())
            ef[9] = float((d_pos - s_pos).mean())
        else:
            if dataset.lower() == "hdfs" and hdfs_feature_contract == "notebook_raw_v1":
                ef = np.array([float(len(s_pos))], dtype=np.float32)
            else:
                ef = np.array([float(np.log1p(len(s_pos)))], dtype=np.float32)

        src_list.append(cid_to_idx[src])
        dst_list.append(cid_to_idx[dst])
        edge_feats_list.append(ef)

    x = torch.from_numpy(node_feats)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    if edge_feats_list:
        edge_feats_np = np.nan_to_num(
            np.array(edge_feats_list, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        edge_feats_np = np.zeros((0, edge_dim), dtype=np.float32)
    edge_attr = torch.from_numpy(edge_feats_np)
    y = torch.tensor([label], dtype=torch.long)

    kwargs = {id_col: seq_id, "num_nodes": num_nodes}
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, **kwargs)
