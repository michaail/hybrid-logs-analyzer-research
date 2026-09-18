"""Train-unit splits used before Drain / TF-IDF (inductive_v1)."""

from __future__ import annotations

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


def bgl_train_line_count(log_path: str | Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> int:
    """Training event count for a BGL file (same cut as the time-split sequencer)."""
    n_train, _, _ = event_count_slices(count_nonempty_lines(log_path), train_ratio, val_ratio)
    return n_train


def save_unit_split(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_unit_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
