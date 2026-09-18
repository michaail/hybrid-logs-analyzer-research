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
    densify_tfidf_rows(matrix) → ndarray
    build_graph_structures(sequences, block_labels, *, ...) → (list[dict], stats)
    attach_cluster_embeddings(structures, cluster_embeddings, *, ...) → list[Data]
    build_pyg_dataset(sequences, block_labels, cluster_embeddings, *, ...) → list[Data]
    split_dataset(all_data, seed, *, ...) → (idx_train, idx_val, idx_test)
    save_graph_dataset(all_data, ..., path, *, ...)
    load_graph_splits(path) → (train, val, test, meta)
    GraphDirectoryDataset(directory) — lazy per-graph ``*.pt`` shards
    save_graph_structures(path, structures, meta)
    load_graph_structures(path) → (structures, meta)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
HdfsFeatureContract = Literal["notebook_raw_v1", "stabilized_v2"]
NODE_EXTRA_DIM = 9
STRUCTURE_EDGE_DIM = 10
STRUCTURE_CACHE_VERSION = 1


class MissingClusterEmbedding(ValueError):
    """Raised when a sequence references a cluster ID absent from frozen embeddings."""

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id = cluster_id
        super().__init__(f"Cluster {cluster_id} has no frozen embedding.")


class MissingSequenceLabel(KeyError):
    """Raised when a sequence has no ground-truth anomaly label."""

    def __init__(self, sequence_id: Any) -> None:
        self.sequence_id = sequence_id
        super().__init__(
            f"Sequence {sequence_id!r} has no label. "
            "Refusing to treat unlabeled sequences as normal."
        )


TFIDF_DENSE_MAX_FEATURES = 10_000


def _require_torch_geometric() -> None:
    """Fail before the per-sequence loop if PyG is missing from this venv."""
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_geometric is required to build graph_dataset.pt. "
            "This repository venv needs the same PyG as the local Torch 2.2.2 "
            "baseline: pip install torch-geometric==2.3.0"
        ) from exc


# ── Embedding computation ─────────────────────────────────────────────────────


def compute_embeddings(
    templates_data: list[dict],
    cluster_to_enriched: dict,
    *,
    tfidf_enabled: bool = True,
    sbert_enabled: bool = True,
    tfidf_fit_texts: list[str] | None = None,
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
    tfidf_fit_texts : list[str] | None
        If given, the TF-IDF vocabulary and IDF are fitted on this subset
        (train templates) and every template is then ``transform``ed.
        ``None`` fits on all templates (transductive).

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
        fit_corpus = list(tfidf_fit_texts) if tfidf_fit_texts is not None else all_templates
        if not fit_corpus:
            raise ValueError("TF-IDF requires at least one training template.")
        tfidf_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s]+")
        tfidf_vectorizer.fit(fit_corpus)
        # Densify the *template table* (tens–hundreds of rows). Safe for HDFS/BGL
        # (vocab ≪ 10k). A much larger vocabulary should stay sparse until each
        # graph materialises its node matrix; see densify_tfidf_rows().
        tfidf_matrix = tfidf_vectorizer.transform(all_templates)
        n_features = int(tfidf_matrix.shape[1])
        if n_features > TFIDF_DENSE_MAX_FEATURES:
            logger.warning(
                "TF-IDF vocabulary has %d features (> %d). Densifying the "
                "template table anyway because GINE still needs dense node "
                "rows; consider a hashed/sparse path for a larger corpus.",
                n_features,
                TFIDF_DENSE_MAX_FEATURES,
            )
        tfidf_dense = densify_tfidf_rows(tfidf_matrix)
        parts.append(tfidf_dense)
        logger.info(
            "TF-IDF: %d-dim vectors for %d templates (fitted on %d)",
            tfidf_dense.shape[1],
            len(all_cids),
            len(fit_corpus),
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


def densify_tfidf_rows(matrix: Any) -> np.ndarray:
    """Materialise TF-IDF rows as float32. Template tables stay small (n ≪ 1k).

    GINEConv expects dense node features, so each template is densified here
    rather than storing a (n_graphs × vocab) cube. Call this per template
    table, not per log line.
    """
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


# ── Graph structure (embedding-agnostic) ──────────────────────────────────────


def build_graph_structures(
    sequences: dict,
    block_labels: dict,
    *,
    dataset: str = "bgl",
    on_graph_error: Literal["skip", "fail"] = "skip",
    hdfs_feature_contract: HdfsFeatureContract = "stabilized_v2",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build collapsed topology + node/edge extras, without template embeddings.

    Family A arms that share ``dataset``, ``unique_sequences``, and
    ``feature_contract`` can reuse this list and splice a different embedding
    matrix. Full 10-d edges are always stored; ``use_edge_features=False`` is a
    later column slice, not a second construction pass.
    """
    structures: list[dict[str, Any]] = []
    skipped = 0
    t0 = time.time()
    from tqdm import tqdm

    iterator = tqdm(
        sequences.items(),
        total=len(sequences),
        desc="Graph structure",
        unit="seq",
        mininterval=2.0,
    )
    for wid, seq in iterator:
        if wid not in block_labels:
            raise MissingSequenceLabel(wid)
        label = int(block_labels[wid])
        try:
            structures.append(
                _seq_to_structure(
                    seq,
                    label,
                    dataset=dataset,
                    hdfs_feature_contract=hdfs_feature_contract,
                )
            )
        except Exception as exc:
            if on_graph_error == "fail":
                raise
            skipped += 1
            if skipped <= 5:
                logger.warning("Skipped sequence %s: %s", wid, exc)

    stats = {"n_graphs": len(structures), "n_skipped": skipped}
    logger.info(
        "Built %d graph structures in %.1fs  (skipped %d)",
        stats["n_graphs"],
        time.time() - t0,
        skipped,
    )
    return structures, stats


def attach_cluster_embeddings(
    structures: list[dict[str, Any]],
    cluster_embeddings: dict[int, np.ndarray],
    *,
    use_edge_features: bool = True,
    missing_embedding: Literal["zero", "fail"] = "zero",
) -> list:
    """Splice template embeddings onto cached structures to produce PyG graphs."""
    _require_torch_geometric()
    import torch
    from torch_geometric.data import Data

    normalized = {
        int(cid): np.asarray(vector, dtype=np.float32)
        for cid, vector in cluster_embeddings.items()
    }
    embed_dim = next(iter(normalized.values())).shape[0]
    if missing_embedding == "fail":
        for structure in structures:
            for cluster_id in structure["cluster_ids"]:
                if int(cluster_id) not in normalized:
                    raise MissingClusterEmbedding(int(cluster_id))

    from tqdm import tqdm

    all_data = []
    t0 = time.time()
    iterator = tqdm(
        structures,
        total=len(structures),
        desc="PyG splice",
        unit="seq",
        mininterval=2.0,
    )
    for structure in iterator:
        node_feats = _node_features_from_structure(
            structure, normalized, embed_dim, missing_embedding
        )
        edge_attr_np = structure["edge_attr"]
        if use_edge_features:
            edge_attr_np = np.asarray(edge_attr_np, dtype=np.float32)
        elif edge_attr_np.shape[0] == 0:
            edge_attr_np = np.zeros((0, 1), dtype=np.float32)
        else:
            edge_attr_np = np.asarray(edge_attr_np[:, :1], dtype=np.float32)

        id_col = structure["id_col"]
        kwargs = {id_col: structure["seq_id"], "num_nodes": int(structure["num_nodes"])}
        all_data.append(
            Data(
                x=torch.from_numpy(node_feats),
                edge_index=torch.from_numpy(np.asarray(structure["edge_index"], dtype=np.int64)),
                edge_attr=torch.from_numpy(edge_attr_np),
                y=torch.tensor([int(structure["y"])], dtype=torch.long),
                **kwargs,
            )
        )
    logger.info("Spliced embeddings onto %d graphs in %.1fs", len(all_data), time.time() - t0)
    return all_data


def save_graph_structures(
    path: str | Path,
    structures: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist embedding-agnostic graph structures for Family A reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STRUCTURE_CACHE_VERSION,
        "structures": structures,
        "meta": meta or {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Graph structures saved → %s  (%.1f MB)", path, path.stat().st_size / 1e6)


def load_graph_structures(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a payload written by :func:`save_graph_structures`."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("version", -1)) != STRUCTURE_CACHE_VERSION:
        raise ValueError(
            f"Unsupported graph structure cache version {payload.get('version')!r} in {path}"
        )
    return list(payload["structures"]), dict(payload.get("meta") or {})


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
    structures, _ = build_graph_structures(
        sequences,
        block_labels,
        dataset=dataset,
        on_graph_error=on_graph_error,
        hdfs_feature_contract=hdfs_feature_contract,
    )
    if not structures:
        return []
    return attach_cluster_embeddings(
        structures,
        cluster_embeddings,
        use_edge_features=use_edge_features,
        missing_embedding=missing_embedding,
    )


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


def split_label_indices(
    labels: np.ndarray,
    seed: int = 42,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 split from a label vector (no PyG graphs required)."""
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    test_ratio = 1.0 - train_ratio - val_ratio
    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=labels[idx_temp],
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
    _save_graph_splits(
        path,
        all_data,
        idx_train,
        idx_val,
        idx_test,
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        dataset_meta=dataset_meta,
    )


def graph_split_dir(path: str | Path) -> Path:
    """Directory of per-split shards next to ``graph_dataset.pt``."""
    path = Path(path)
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    stem = Path(name).stem
    return path.parent / f"{stem}_splits"


def _save_graph_splits(
    path: Path,
    all_data: list,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    node_dim: int,
    edge_dim: int,
    embed_dim: int,
    dataset_meta: dict[str, Any] | None,
) -> None:
    """Write train/val/test lists so training need not ``torch.load`` every graph."""
    import torch

    split_dir = graph_split_dir(path)
    split_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "embed_dim": embed_dim,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_total": int(len(all_data)),
    }
    if dataset_meta:
        meta["dataset_meta"] = dataset_meta
    torch.save([all_data[int(i)] for i in idx_train], split_dir / "train.pt")
    torch.save([all_data[int(i)] for i in idx_val], split_dir / "val.pt")
    torch.save([all_data[int(i)] for i in idx_test], split_dir / "test.pt")
    (split_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Graph splits saved → %s", split_dir)


class GraphDirectoryDataset:
    """Lazy ``__getitem__`` over one graph per ``*.pt`` file.

    Unique-sequence HDFS (~17.6k) already avoids loading 575k blocks. Use this
    only when training the full block set: write shards with
    :func:`save_graph_directory`, then wrap each split directory. Metrics are
    unchanged if the graphs themselves are identical.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No *.pt graphs in {self.directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        return torch.load(self.paths[index], weights_only=False, map_location="cpu")


def save_graph_directory(graphs: list, directory: str | Path) -> Path:
    """Write one ``{index:06d}.pt`` per graph for :class:`GraphDirectoryDataset`."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, graph in enumerate(graphs):
        torch.save(graph, directory / f"{index:06d}.pt")
    return directory


def load_graph_splits(path: str | Path) -> tuple[list, list, list, dict[str, Any]]:
    """Load train/val/test graphs without materialising the full list when shards exist.

    Falls back to a single ``torch.load`` of ``data_list`` for older bundles.
    Unique-sequence campaigns load three split lists (~17.6k total). A 575k
    all-block run should use :class:`GraphDirectoryDataset` instead of one
    ``torch.save`` list.
    """
    import torch

    path = Path(path)
    split_dir = graph_split_dir(path)
    train_path = split_dir / "train.pt"
    if train_path.exists() and (split_dir / "val.pt").exists() and (split_dir / "test.pt").exists():
        train_graphs = torch.load(train_path, weights_only=False, map_location="cpu")
        val_graphs = torch.load(split_dir / "val.pt", weights_only=False, map_location="cpu")
        test_graphs = torch.load(split_dir / "test.pt", weights_only=False, map_location="cpu")
        meta: dict[str, Any] = {}
        meta_path = split_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        sidecar = path.with_name("dataset_meta.json")
        if sidecar.exists():
            meta.setdefault("dataset_meta", json.loads(sidecar.read_text()))
        return list(train_graphs), list(val_graphs), list(test_graphs), meta

    bundle = torch.load(path, weights_only=False, map_location="cpu")
    all_data = bundle["data_list"]
    train_graphs = [all_data[int(i)] for i in bundle["idx_train"]]
    val_graphs = [all_data[int(i)] for i in bundle["idx_val"]]
    test_graphs = [all_data[int(i)] for i in bundle["idx_test"]]
    meta = {
        "node_dim": bundle.get("node_dim"),
        "edge_dim": bundle.get("edge_dim"),
        "embed_dim": bundle.get("embed_dim"),
        "dataset_meta": bundle.get("dataset_meta") or {},
    }
    return train_graphs, val_graphs, test_graphs, meta


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_numeric(x: object) -> bool:
    """Return True only for finite non-NaN/Inf numeric values."""
    try:
        v = float(cast(Any, x))
        return np.isfinite(v)
    except (ValueError, TypeError):
        return False


def _seq_to_structure(
    seq: pd.DataFrame,
    label: int,
    *,
    dataset: str,
    hdfs_feature_contract: HdfsFeatureContract = "stabilized_v2",
) -> dict[str, Any]:
    """Collapsed topology, 9-d node extras, and 10-d edges for one sequence."""
    ts_col = "unix_ts" if dataset.lower() == "bgl" else "timestamp"
    id_col = "window_id" if dataset.lower() == "bgl" else "block_id"

    seq_id = seq[id_col].iloc[0]
    cids = seq["cluster_id"].tolist()
    params = seq["parameters"].tolist() if "parameters" in seq.columns else [[] for _ in cids]
    ts = seq[ts_col].tolist() if ts_col in seq.columns else [None] * len(cids)
    n = len(cids)

    node_params: dict = defaultdict(list)
    node_count = Counter(cids)
    node_positions: dict = defaultdict(list)

    for i, (cid, p) in enumerate(zip(cids, params)):
        node_params[cid].extend(p if isinstance(p, list) else [])
        node_positions[cid].append(i / max(n - 1, 1))

    unique_cids = [int(cid) for cid in dict.fromkeys(cids)]
    cid_to_idx = {cid: idx for idx, cid in enumerate(unique_cids)}
    num_nodes = len(unique_cids)
    raw_hdfs = dataset.lower() == "hdfs" and hdfs_feature_contract == "notebook_raw_v1"

    node_extra = np.zeros((num_nodes, NODE_EXTRA_DIM), dtype=np.float32)
    for idx, cid in enumerate(unique_cids):
        nums = [float(x) for x in node_params[cid] if _is_numeric(x)]
        pos = np.array(node_positions[cid])
        if raw_hdfs:
            node_extra[idx, 0] = float(node_count[cid])
            node_extra[idx, 1] = float(len(node_params[cid]))
            node_extra[idx, 2] = float(np.mean(nums)) if nums else 0.0
            node_extra[idx, 3] = float(np.max(nums)) if nums else 0.0
        else:
            node_extra[idx, 0] = float(np.log1p(node_count[cid]))
            node_extra[idx, 1] = float(np.log1p(len(node_params[cid])))
            node_extra[idx, 2] = float(np.log1p(abs(np.mean(nums)))) if nums else 0.0
            node_extra[idx, 3] = float(np.log1p(abs(np.max(nums)))) if nums else 0.0
        node_extra[idx, 4] = float(pos.min())
        node_extra[idx, 5] = float(pos.max())
        node_extra[idx, 6] = float(pos.mean())
        node_extra[idx, 7] = float(pos.std()) if len(pos) > 1 else 0.0
        node_extra[idx, 8] = float(pos.max() - pos.min())

    node_extra = np.nan_to_num(node_extra, nan=0.0, posinf=0.0, neginf=0.0)

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
                if raw_hdfs:
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
        ef = np.zeros(STRUCTURE_EDGE_DIM, dtype=np.float32)
        if raw_hdfs:
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
        src_list.append(cid_to_idx[int(src)])
        dst_list.append(cid_to_idx[int(dst)])
        edge_feats_list.append(ef)

    if edge_feats_list:
        edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
        edge_attr = np.nan_to_num(
            np.asarray(edge_feats_list, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, STRUCTURE_EDGE_DIM), dtype=np.float32)

    return {
        "cluster_ids": np.asarray(unique_cids, dtype=np.int64),
        "node_extra": node_extra,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": int(label),
        "id_col": id_col,
        "seq_id": seq_id,
        "num_nodes": num_nodes,
    }


def _node_features_from_structure(
    structure: dict[str, Any],
    cluster_embeddings: dict[int, np.ndarray],
    embed_dim: int,
    missing_embedding: Literal["zero", "fail"],
) -> np.ndarray:
    cluster_ids = [int(cid) for cid in structure["cluster_ids"]]
    num_nodes = len(cluster_ids)
    node_feats = np.zeros((num_nodes, embed_dim + NODE_EXTRA_DIM), dtype=np.float32)
    for idx, cid in enumerate(cluster_ids):
        if cid not in cluster_embeddings:
            if missing_embedding == "fail":
                raise MissingClusterEmbedding(cid)
            emb = np.zeros(embed_dim, dtype=np.float32)
        else:
            emb = np.asarray(cluster_embeddings[cid], dtype=np.float32)
        node_feats[idx, :embed_dim] = emb
    node_feats[:, embed_dim:] = structure["node_extra"]
    return np.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)


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
    del node_dim, edge_dim  # derived from embeddings + cached 10-d structure
    structure = _seq_to_structure(
        seq,
        label,
        dataset=dataset,
        hdfs_feature_contract=hdfs_feature_contract,
    )
    return attach_cluster_embeddings(
        [structure],
        cluster_embeddings,
        use_edge_features=use_edge_features,
        missing_embedding=missing_embedding,
    )[0]
