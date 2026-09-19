"""Stage 3 — Log-line sequencer.

Two strategies, toggled via ``dataset``:

* ``"hdfs"`` — Block-ID grouping: every log line that references a block
  is grouped with all other lines sharing the same ``block_id``, producing
  one sequence per HDFS block (one sequence = one anomaly-labelling unit).

* ``"bgl"`` — Sliding time-window: the log is partitioned into time windows.
  A window is labelled anomalous if at least one of its lines has
  ``is_anomaly == True``.

  ``split="time"`` cuts the event stream into train/val/test **before**
  window construction so a window cannot straddle the cut and overlap is
  confined to one split. ``split="stratified"`` windows the full stream
  (notebook-era behaviour; pair only with a later random graph split).

Public API
----------
    build_sequences(df, dataset, **kwargs)  → dict
    save_sequences(df_blocks, output_path)
    load_sequences(path, dataset)           → (df_blocks, sequences)
    event_count_slices(n, train_ratio, val_ratio)
    bgl_split_name(window_id)
"""

from __future__ import annotations

import json
import logging
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Namespace window_id by split so train/val/test unix starts cannot collide.
BGL_SPLIT_STRIDE = 1 << 40


def _parquet_engine() -> str:
    """Prefer fastparquet for legacy artifacts, with a PyArrow fallback."""
    return "fastparquet" if find_spec("fastparquet") else "pyarrow"


def event_count_slices(
    n: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) covering *n* time-ordered events."""
    if n <= 0:
        return 0, 0, 0
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_test = 1 if n >= 3 else 0
        leftover = n - n_test
        n_train = max(1, leftover // 2) if leftover >= 2 else leftover
        n_val = leftover - n_train
        return n_train, n_val, n - n_train - n_val
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def bgl_split_name(window_id: int) -> str:
    """Map a namespaced BGL window_id back to train/val/test."""
    value = int(window_id)
    if value >= 2 * BGL_SPLIT_STRIDE:
        return "test"
    if value >= BGL_SPLIT_STRIDE:
        return "val"
    return "train"


def split_indices_from_bgl_windows(sequence_ids: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build idx_train/val/test from time-namespaced window ids (stable order)."""
    train, val, test = [], [], []
    for index, sequence_id in enumerate(sequence_ids):
        name = bgl_split_name(int(sequence_id))
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


def build_sequences(
    df: pd.DataFrame,
    dataset: str,
    *,
    window_minutes: int = 20,
    step_minutes: int = 10,
    split: str = "stratified",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    embargo_minutes: int = 0,
) -> dict:
    """Route to the dataset-appropriate sequencer.

    Parameters
    ----------
    df : pd.DataFrame
        Annotated log DataFrame produced by the parser stage.
    dataset : str
        ``"hdfs"`` or ``"bgl"``.
    window_minutes, step_minutes : int
        BGL-only sliding-window parameters (ignored for HDFS).
    split : str
        BGL only: ``"time"`` (cut events, then window) or ``"stratified"``
        (window the full stream).
    train_ratio, val_ratio : float
        Event-count fractions used when ``split="time"``.
    embargo_minutes : int
        BGL time only: discard events within this many minutes on either
        side of each chronological partition boundary.

    Returns
    -------
    dict
        ``{sequence_id: group_DataFrame}`` mapping.  For HDFS the keys are
        ``block_id`` strings; for BGL they are integer window ids.
    """
    dataset = dataset.lower()
    if dataset == "hdfs":
        return _build_hdfs_sequences(df)
    if dataset == "bgl":
        if str(split).lower() == "time":
            return _build_bgl_sequences_time_split(
                df,
                window_minutes=window_minutes,
                step_minutes=step_minutes,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                embargo_minutes=embargo_minutes,
            )
        return _build_bgl_sequences(df, window_minutes=window_minutes, step_minutes=step_minutes)
    raise ValueError(f"Unknown dataset {dataset!r}. Choose 'hdfs' or 'bgl'.")


def save_sequences(df_blocks: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the flat sequences DataFrame to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_save = df_blocks.copy()
    if "parameters" in df_save.columns:
        df_save["parameters"] = df_save["parameters"].apply(json.dumps)
    for col in df_save.select_dtypes(include=["object", "string"]).columns:
        df_save[col] = df_save[col].astype(object)
    df_save.to_parquet(output_path, index=False, engine=_parquet_engine())
    logger.info("Saved %d rows → %s", len(df_save), output_path)


def load_sequences(
    path: str | Path,
    dataset: str,
) -> tuple[pd.DataFrame, dict]:
    """Load a sequences Parquet file and reconstruct the ``{id → group}`` dict.

    Returns
    -------
    df_blocks : pd.DataFrame
    sequences : dict
    """
    path = Path(path)
    df_blocks = pd.read_parquet(path, engine=_parquet_engine())
    if "parameters" in df_blocks.columns:
        df_blocks["parameters"] = df_blocks["parameters"].apply(json.loads)

    id_col = "window_id" if dataset.lower() == "bgl" else "block_id"
    sequences = {sid: grp for sid, grp in df_blocks.groupby(id_col, sort=False)}
    return df_blocks, sequences


def _build_hdfs_sequences(df: pd.DataFrame) -> dict:
    """Group HDFS log lines by ``block_id`` (one sequence per HDFS block)."""
    t0 = time.time()
    df_blocks = df.loc[df["block_id"].notna()].sort_values(["block_id", "timestamp"])
    sequences = {
        block_id: group for block_id, group in df_blocks.groupby("block_id", sort=False)
    }
    n_dropped = len(df) - len(df_blocks)
    logger.info(
        "HDFS sequencer: %d blocks in %.2fs  (dropped %d lines without block_id)",
        len(sequences),
        time.time() - t0,
        n_dropped,
    )
    return sequences


def _window_one_span(
    df: pd.DataFrame,
    *,
    window_minutes: int,
    step_minutes: int,
    id_offset: int,
) -> dict:
    """Sliding windows over one contiguous time span (one split)."""
    if df.empty:
        return {}
    df = df.dropna(subset=["unix_ts"]).sort_values("unix_ts").reset_index(drop=True)
    df["unix_ts"] = df["unix_ts"].astype(np.int64)
    window_seconds = window_minutes * 60
    step_seconds = step_minutes * 60
    unix_arr = df["unix_ts"].values
    t_min = int(unix_arr[0])
    t_max = int(unix_arr[-1])
    if t_max < t_min + window_seconds:
        # Span shorter than one window: keep a single window of whatever is there.
        window_starts = np.array([t_min], dtype=np.int64)
    else:
        window_starts = np.arange(
            t_min, t_max - window_seconds + 1, step_seconds, dtype=np.int64
        )

    row_indices: list[np.ndarray] = []
    win_ids: list[np.ndarray] = []
    for w_start in window_starts:
        lo = int(np.searchsorted(unix_arr, w_start, side="left"))
        hi = int(np.searchsorted(unix_arr, w_start + window_seconds, side="left"))
        n = hi - lo
        if n == 0:
            continue
        wid = int(w_start) + int(id_offset)
        row_indices.append(np.arange(lo, hi, dtype=np.int64))
        win_ids.append(np.full(n, wid, dtype=np.int64))

    if not row_indices:
        return {}
    all_rows = np.concatenate(row_indices)
    all_wids = np.concatenate(win_ids)
    df_windows = df.iloc[all_rows].copy()
    df_windows["window_id"] = all_wids
    df_windows = df_windows.sort_values(["window_id", "unix_ts"]).reset_index(drop=True)
    return {wid: grp for wid, grp in df_windows.groupby("window_id", sort=False)}


def _build_bgl_sequences(
    df: pd.DataFrame,
    *,
    window_minutes: int = 20,
    step_minutes: int = 10,
) -> dict:
    """Partition BGL log lines into sliding time windows on the full stream."""
    t0 = time.time()
    sequences = _window_one_span(
        df, window_minutes=window_minutes, step_minutes=step_minutes, id_offset=0
    )
    if not sequences:
        logger.warning("BGL sequencer produced 0 non-empty windows. Check log timestamps.")
        return {}
    logger.info(
        "BGL sequencer: %d non-empty windows (W=%d min, step=%d min) in %.2fs",
        len(sequences),
        window_minutes,
        step_minutes,
        time.time() - t0,
    )
    return sequences


def _build_bgl_sequences_time_split(
    df: pd.DataFrame,
    *,
    window_minutes: int,
    step_minutes: int,
    train_ratio: float,
    val_ratio: float,
    embargo_minutes: int = 0,
) -> dict:
    """Cut the event stream by count, embargo boundaries, then window each split."""
    t0 = time.time()
    ordered = df.dropna(subset=["unix_ts"]).sort_values("unix_ts").reset_index(drop=True)
    n_train, n_val, n_test = event_count_slices(len(ordered), train_ratio, val_ratio)
    embargo_seconds = int(embargo_minutes) * 60
    if embargo_seconds < 0:
        raise ValueError("embargo_minutes must be non-negative.")
    train_cut = int(ordered["unix_ts"].iloc[n_train - 1])
    val_cut = int(ordered["unix_ts"].iloc[n_train + n_val - 1])
    train_span = ordered.iloc[:n_train]
    val_span = ordered.iloc[n_train : n_train + n_val]
    test_span = ordered.iloc[n_train + n_val :]
    if embargo_seconds:
        train_span = train_span.loc[train_span["unix_ts"] <= train_cut - embargo_seconds]
        val_span = val_span.loc[
            (val_span["unix_ts"] > train_cut + embargo_seconds)
            & (val_span["unix_ts"] <= val_cut - embargo_seconds)
        ]
        test_span = test_span.loc[test_span["unix_ts"] > val_cut + embargo_seconds]
    if train_span.empty or val_span.empty or test_span.empty:
        raise ValueError(
            "BGL time split has an empty partition after applying the "
            f"{embargo_minutes}-minute embargo."
        )
    slices = {
        "train": (train_span, 0),
        "val": (val_span, BGL_SPLIT_STRIDE),
        "test": (test_span, 2 * BGL_SPLIT_STRIDE),
    }
    sequences: dict = {}
    counts: dict[str, int] = {}
    for name, (span, offset) in slices.items():
        part = _window_one_span(
            span, window_minutes=window_minutes, step_minutes=step_minutes, id_offset=offset
        )
        counts[name] = len(part)
        sequences.update(part)
    logger.info(
        "BGL time-split sequencer: train=%d val=%d test=%d windows "
        "(events %d/%d/%d before embargo, embargo=%d min, W=%d min, step=%d min) in %.2fs",
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
        n_train,
        n_val,
        n_test,
        embargo_minutes,
        window_minutes,
        step_minutes,
        time.time() - t0,
    )
    return sequences
