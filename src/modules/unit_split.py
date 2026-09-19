"""Train-unit splits used before Drain / TF-IDF (inductive_v1)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.modules.parser.drain_parser import DrainParser
from src.modules.sequencer import event_count_slices


def scan_hdfs_block_ids(log_path: str | Path) -> list[str]:
    """Return unique HDFS block ids in first-seen order (regex, no Drain)."""
    seen: set[str] = set()
    ordered: list[str] = []
    with Path(log_path).open("r", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            block_id = DrainParser.extract_hdfs_block_id(line)
            if block_id and block_id not in seen:
                seen.add(block_id)
                ordered.append(block_id)
    return ordered


def count_nonempty_lines(log_path: str | Path) -> int:
    """Count non-empty lines (BGL event-count cuts follow file order)."""
    count = 0
    with Path(log_path).open("r", errors="replace") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def stratified_id_split(
    ids: list,
    labels: dict[Any, int],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list]:
    """70/15/15 stratified split of labelled ids."""
    from sklearn.model_selection import train_test_split

    missing = [item for item in ids if item not in labels]
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise KeyError(
            f"{len(missing)} unit(s) have no label (e.g. {preview}). "
            "Refusing to treat unlabeled sequences as normal."
        )
    y = np.array([int(labels[item]) for item in ids])
    indices = np.arange(len(ids))
    test_ratio = 1.0 - train_ratio - val_ratio
    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y,
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=y[idx_temp],
    )
    array_ids = np.asarray(ids, dtype=object)
    return {
        "train": [array_ids[int(i)] for i in idx_train],
        "val": [array_ids[int(i)] for i in idx_val],
        "test": [array_ids[int(i)] for i in idx_test],
    }


def topology_grouped_id_split(
    ids: list,
    labels: dict[Any, int],
    fingerprints: dict[Any, str],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list]:
    """Split IDs while keeping every topology fingerprint in one partition.

    Repeated ``GroupShuffleSplit`` candidates keep the rare anomaly class
    reasonably balanced without leaking an ordered-template fingerprint.
    The returned units remain full HDFS blocks; no topology is deduplicated.
    """
    from sklearn.model_selection import GroupShuffleSplit

    missing_labels = [item for item in ids if item not in labels]
    missing_groups = [item for item in ids if item not in fingerprints]
    if missing_labels or missing_groups:
        raise KeyError(
            f"Topology split requires labels and fingerprints for every unit "
            f"(missing labels={len(missing_labels)}, groups={len(missing_groups)})."
        )
    y = np.asarray([int(labels[item]) for item in ids], dtype=np.int64)
    groups = np.asarray([str(fingerprints[item]) for item in ids], dtype=object)
    indices = np.arange(len(ids), dtype=np.int64)
    array_ids = np.asarray(ids, dtype=object)

    def best_group_split(
        candidate_indices: np.ndarray,
        candidate_y: np.ndarray,
        candidate_groups: np.ndarray,
        test_size: float,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_rate = float(candidate_y.mean())
        splitter = GroupShuffleSplit(
            n_splits=64, test_size=test_size, random_state=random_state
        )
        candidates = list(
            splitter.split(candidate_indices, candidate_y, candidate_groups)
        )
        return min(
            candidates,
            key=lambda pair: (
                abs(len(pair[1]) / len(candidate_indices) - test_size)
                + abs(float(candidate_y[pair[1]].mean()) - target_rate)
            ),
        )

    idx_train_base, idx_holdout = best_group_split(
        indices, y, groups, 1.0 - train_ratio, seed
    )
    holdout_groups = groups[idx_holdout]
    holdout_y = y[idx_holdout]
    test_ratio = 1.0 - train_ratio - val_ratio
    inner_train, inner_test = best_group_split(
        idx_holdout,
        holdout_y,
        holdout_groups,
        test_ratio / (val_ratio + test_ratio),
        seed + 1,
    )
    idx_val = idx_holdout[inner_train]
    idx_test = idx_holdout[inner_test]

    # Keep the requested naming and guarantee exhaustive, disjoint membership.
    result = {
        "train": [array_ids[int(i)] for i in idx_train_base],
        "val": [array_ids[int(i)] for i in idx_val],
        "test": [array_ids[int(i)] for i in idx_test],
    }
    split_groups = [
        {str(fingerprints[item]) for item in result[name]}
        for name in ("train", "val", "test")
    ]
    if split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2]:
        raise RuntimeError("Topology-grouped split leaked a fingerprint across partitions.")
    return result


def hdfs_topology_fingerprints(frame: Any) -> dict[str, str]:
    """Hash each block's ordered provisional Drain cluster sequence."""
    required = {"block_id", "cluster_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Annotated HDFS frame must contain {sorted(required)}.")
    fingerprints: dict[str, str] = {}
    for block_id, group in frame.groupby("block_id", sort=False):
        payload = "|".join(str(int(value)) for value in group["cluster_id"].tolist())
        fingerprints[str(block_id)] = hashlib.sha256(payload.encode()).hexdigest()
    return fingerprints


def indices_from_unit_split(
    sequence_ids: list[str],
    unit_split: dict[str, list],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map graph sequence ids onto a frozen unit split (block_id membership)."""
    membership = {}
    for name in ("train", "val", "test"):
        for item in unit_split.get(name, []):
            membership[str(item)] = name
    missing = [sid for sid in sequence_ids if str(sid) not in membership]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} graph id(s) are not in the unit split (e.g. {preview})."
        )
    train, val, test = [], [], []
    for index, sequence_id in enumerate(sequence_ids):
        name = membership[str(sequence_id)]
        if name == "train":
            train.append(index)
        elif name == "val":
            val.append(index)
        else:
            test.append(index)
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def hdfs_train_line_filter(train_blocks: set[str]) -> Callable[[str], bool]:
    """Keep Drain-fit lines whose block id is in the training split."""

    def include(line: str) -> bool:
        block_id = DrainParser.extract_hdfs_block_id(line)
        return block_id is not None and block_id in train_blocks

    return include


def bgl_first_n_line_filter(n_train: int) -> Callable[[str], bool]:
    """Keep the first *n_train* nonempty lines (file/time order on BGL)."""
    state = {"seen": 0}

    def include(_line: str) -> bool:
        state["seen"] += 1
        return state["seen"] <= n_train

    return include


def bgl_time_split_boundaries(
    log_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[int, int]:
    """Return BGL event-time cut timestamps used by the sequencer.

    Invalid or empty records are excluded just as they are in the BGL time
    sequencer. Sorting timestamps rather than relying on file order keeps the
    parser-fit boundary identical to the graph split boundary.
    """
    timestamps: list[int] = []
    with Path(log_path).open("r", errors="replace") as handle:
        for raw in handle:
            parts = raw.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                timestamps.append(int(parts[1]))
            except ValueError:
                continue
    timestamps.sort()
    n_train, n_val, n_test = event_count_slices(len(timestamps), train_ratio, val_ratio)
    if not n_train or not n_val or not n_test:
        raise ValueError("BGL requires non-empty train, validation, and test time partitions.")
    return timestamps[n_train - 1], timestamps[n_train + n_val - 1]


def bgl_train_time_filter(train_end_timestamp: int) -> Callable[[str], bool]:
    """Keep BGL lines at or before the retained chronological train boundary."""

    def include(line: str) -> bool:
        parts = line.split(None, 2)
        try:
            return len(parts) >= 2 and int(parts[1]) <= train_end_timestamp
        except ValueError:
            return False

    return include


def bgl_train_line_count(log_path: str | Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> int:
    """Training event count for a BGL file (same cut as the time-split sequencer)."""
    n_train, _, _ = event_count_slices(count_nonempty_lines(log_path), train_ratio, val_ratio)
    return n_train


def save_unit_split(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_unit_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
