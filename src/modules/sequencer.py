"""Stage 3 — Log-line sequencer.

Two strategies, toggled via ``dataset``:

* ``"hdfs"`` — Block-ID grouping: every log line that references a block
  is grouped with all other lines sharing the same ``block_id``, producing
  one sequence per HDFS block (one sequence = one anomaly-labelling unit).

* ``"bgl"`` — Sliding time-window: the log is partitioned into overlapping
  fixed-length time windows.  Every line falling inside a window appears in
  that window's sequence.  A window is labelled anomalous if at least one
  of its lines has ``is_anomaly == True``.

Public API
----------
    build_sequences(df, dataset, **kwargs)  → dict
    save_sequences(df_blocks, output_path)
    load_sequences(path, dataset)           → (df_blocks, sequences)
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


def _parquet_engine() -> str:
    """Prefer fastparquet for legacy artifacts, with a PyArrow fallback."""
    return "fastparquet" if find_spec("fastparquet") else "pyarrow"


# ── Public API ────────────────────────────────────────────────────────────────


def build_sequences(
    df: pd.DataFrame,
    dataset: str,
    *,
    window_minutes: int = 20,
    step_minutes: int = 10,
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

    Returns
    -------
    dict
        ``{sequence_id: group_DataFrame}`` mapping.  For HDFS the keys are
        ``block_id`` strings; for BGL they are integer UNIX window-start
        timestamps.
    """
    dataset = dataset.lower()
    if dataset == "hdfs":
        return _build_hdfs_sequences(df)
    elif dataset == "bgl":
        return _build_bgl_sequences(df, window_minutes=window_minutes, step_minutes=step_minutes)
    else:
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


# ── HDFS sequencer ────────────────────────────────────────────────────────────


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


# ── BGL sliding-window sequencer ──────────────────────────────────────────────


def _build_bgl_sequences(
    df: pd.DataFrame,
    *,
    window_minutes: int = 20,
    step_minutes: int = 10,
) -> dict:
    """Partition BGL log lines into overlapping sliding time windows.

    Uses ``searchsorted`` for O(W_count × log N) assignment — fast even on
    millions of log lines.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``unix_ts`` column (integer or float seconds since epoch).
    window_minutes : int
        Width of each time window in minutes.
    step_minutes : int
        Stride between consecutive window starts.  ``step < window`` gives
        overlapping windows (recommended for anomaly detection).

    Returns
    -------
    dict
        ``{window_start_unix: group_DataFrame}``
    """
    t0 = time.time()

    df = df.dropna(subset=["unix_ts"]).sort_values("unix_ts").reset_index(drop=True)
    df["unix_ts"] = df["unix_ts"].astype(np.int64)

    window_seconds = window_minutes * 60
    step_seconds = step_minutes * 60

    unix_arr = df["unix_ts"].values
    t_min = int(unix_arr[0])
    t_max = int(unix_arr[-1])

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
        row_indices.append(np.arange(lo, hi, dtype=np.int64))
        win_ids.append(np.full(n, w_start, dtype=np.int64))

    if not row_indices:
        logger.warning("BGL sequencer produced 0 non-empty windows. Check log timestamps.")
        return {}

    all_rows = np.concatenate(row_indices)
    all_wids = np.concatenate(win_ids)

    df_windows = df.iloc[all_rows].copy()
    df_windows["window_id"] = all_wids
    df_windows = df_windows.sort_values(["window_id", "unix_ts"]).reset_index(drop=True)

    sequences = {wid: grp for wid, grp in df_windows.groupby("window_id", sort=False)}
    logger.info(
        "BGL sequencer: %d non-empty windows (W=%d min, step=%d min) in %.2fs",
        len(sequences),
        window_minutes,
        step_minutes,
        time.time() - t0,
    )
    return sequences
