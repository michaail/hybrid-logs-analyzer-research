"""Run tracking utilities for the hybrid log analyzer.

Usage in notebooks
------------------
from src.modules.run_tracker import RunTracker

# 1. At notebook startup — create or open an existing run
tracker = RunTracker.init(
    run_tag=RUN_TAG,
    dataset="HDFS",
    changelog="What changed in this run.",
    justification="Why this run exists.",
    depends_on=["20260407_2026"],          # run_tags this run consumes artifacts from
    pipeline_stage="1_Parser.ipynb",
)

# 2. After writing an artifact
tracker.add_artifact("processed", "data/processed/HDFS/20260407_2026_hdfs_templates.json")
tracker.add_artifact("models",    "models/20260407_2026_attribute_gae.pt")

# 3. After dataset preparation
tracker.set_dataset_stats({
    "total_blocks": 575061,
    "total_anomalous": 16838,
    ...
})

# 4. After model validation / test evaluation
tracker.set_model_metrics("gae_main", {
    "notebook": "6_GAE_Training.ipynb",
    "model_file": "models/20260407_2026_attribute_gae.pt",
    "architecture": {"hidden_dim": 128, "latent_dim": 64},
    "training": {"epochs": 25, "batch_size": 256, "learning_rate": 0.01},
    "val":  {"best_threshold": 0.147, "f1": 0.95, "pr_auc": 0.97, "roc_auc": 0.98,
             "precision": 0.94, "recall": 0.96},
    "test": {"f1": 0.94, "pr_auc": 0.96, "roc_auc": 0.97,
             "precision": 0.93, "recall": 0.95, "confusion_matrix": [[tn, fp], [fn, tp]]},
})
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Use the explicit artifact workspace when provided (for example in Colab).
# Fall back to the repository root rather than ``src/runs``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path(os.environ.get("PIPELINE_WORKSPACE_ROOT", _PROJECT_ROOT))
_RUNS_DIR = _WORKSPACE_ROOT / "runs"


def _run_path(run_tag: str) -> Path:
    return _RUNS_DIR / f"{run_tag}.json"


def _index_path() -> Path:
    return _RUNS_DIR / "index.json"


def _load_index() -> dict:
    path = _index_path()
    if path.exists():
        return json.loads(path.read_text())
    return {"_comment": "Quick-reference index of all experiment runs.", "runs": []}


def _save_index(index: dict) -> None:
    _index_path().write_text(json.dumps(index, indent=2))


def _load_run(run_tag: str) -> dict | None:
    path = _run_path(run_tag)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_run(data: dict) -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _run_path(data["run_tag"]).write_text(json.dumps(data, indent=2))


class RunTracker:
    """Manages a single run file and keeps the shared index in sync."""

    def __init__(self, data: dict) -> None:
        self._data = data

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def init(
        cls,
        run_tag: str,
        dataset: str,
        changelog: str,
        justification: str,
        depends_on: list[str] | None = None,
        pipeline_stage: str | None = None,
    ) -> "RunTracker":
        """Create a new run file or open an existing one.

        If the run file already exists, only ``changelog``, ``justification``,
        and ``pipeline_stage`` are updated so that re-running a notebook does
        not overwrite accumulated artifacts or metrics.
        """
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        existing = _load_run(run_tag)

        if existing is not None:
            existing["changelog"] = changelog
            existing["justification"] = justification
            if pipeline_stage and not any(
                s["notebook"] == pipeline_stage
                for s in existing.get("pipeline_stages", [])
            ):
                existing.setdefault("pipeline_stages", []).append(
                    {"notebook": pipeline_stage, "description": ""}
                )
            tracker = cls(existing)
            tracker._persist()
            print(f"[RunTracker] Opened existing run: {run_tag}")
            return tracker

        date_str = datetime.strptime(run_tag[:8], "%Y%m%d").strftime("%Y-%m-%d")
        data: dict[str, Any] = {
            "run_tag": run_tag,
            "date": date_str,
            "dataset": dataset,
            "changelog": changelog,
            "justification": justification,
            "pipeline_stages": (
                [{"notebook": pipeline_stage, "description": ""}]
                if pipeline_stage
                else []
            ),
            "input_artifacts": _resolve_input_artifacts(depends_on or []),
            "output_artifacts": {"processed": [], "models": []},
            "results": {
                "dataset_stats": {},
                "model_metrics": {},
            },
        }
        tracker = cls(data)
        tracker._persist()
        tracker._update_index(depends_on or [])
        print(f"[RunTracker] Created new run: {run_tag}")
        return tracker

    @classmethod
    def open(cls, run_tag: str) -> "RunTracker":
        """Load an existing run file for updates."""
        data = _load_run(run_tag)
        if data is None:
            raise FileNotFoundError(
                f"No run file found for tag '{run_tag}'. "
                "Use RunTracker.init() to create it first."
            )
        return cls(data)

    # ------------------------------------------------------------------ #
    #  Artifact tracking                                                   #
    # ------------------------------------------------------------------ #

    def add_artifact(self, kind: str, path: str | Path) -> None:
        """Register an output artifact.

        Args:
            kind: ``"processed"`` or ``"models"``.
            path: Workspace-relative path to the artifact file.
        """
        path = str(path)
        bucket: list = self._data["output_artifacts"].setdefault(kind, [])
        if path not in bucket:
            bucket.append(path)
            self._persist()
            print(f"[RunTracker] Artifact registered ({kind}): {path}")

    def add_pipeline_stage(self, notebook: str, description: str = "") -> None:
        """Record that a pipeline notebook has been executed for this run."""
        stages: list = self._data.setdefault("pipeline_stages", [])
        if not any(s["notebook"] == notebook for s in stages):
            stages.append({"notebook": notebook, "description": description})
            self._persist()

    # ------------------------------------------------------------------ #
    #  Results                                                             #
    # ------------------------------------------------------------------ #

    def set_dataset_stats(self, stats: dict) -> None:
        """Overwrite dataset statistics (call after 5_PrepareDataset)."""
        self._data["results"]["dataset_stats"] = stats
        self._persist()
        print(f"[RunTracker] Dataset stats updated for run: {self._data['run_tag']}")

    def set_model_metrics(self, model_key: str, metrics: dict) -> None:
        """Set or replace metrics for one model variant.

        Args:
            model_key: Arbitrary identifier, e.g. ``"gae_main"`` or ``"gae_lr001"``.
            metrics:   Dict with keys ``notebook``, ``model_file``, ``architecture``,
                       ``training``, ``val``, ``test``.
        """
        self._data["results"]["model_metrics"][model_key] = metrics
        self._persist()
        print(
            f"[RunTracker] Model metrics updated: {model_key} "
            f"(run: {self._data['run_tag']})"
        )

    def update_val_metrics(self, model_key: str, **kwargs: Any) -> None:
        """Partially update validation metrics for an existing model entry.

        Example::

            tracker.update_val_metrics("gae_main",
                best_threshold=0.147, f1=0.95, pr_auc=0.97, roc_auc=0.98)
        """
        self._data["results"]["model_metrics"].setdefault(model_key, {}).setdefault(
            "val", {}
        ).update(kwargs)
        self._persist()

    def update_test_metrics(self, model_key: str, **kwargs: Any) -> None:
        """Partially update test metrics for an existing model entry.

        Example::

            tracker.update_test_metrics("gae_main",
                f1=0.94, pr_auc=0.96, roc_auc=0.97,
                precision=0.93, recall=0.95,
                confusion_matrix=[[tn, fp], [fn, tp]])
        """
        self._data["results"]["model_metrics"].setdefault(model_key, {}).setdefault(
            "test", {}
        ).update(kwargs)
        self._persist()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        _save_run(self._data)

    def _update_index(self, depends_on: list[str]) -> None:
        index = _load_index()
        run_tag = self._data["run_tag"]
        if not any(r["run_tag"] == run_tag for r in index["runs"]):
            index["runs"].append(
                {
                    "run_tag": run_tag,
                    "date": self._data["date"],
                    "dataset": self._data["dataset"],
                    "depends_on": depends_on,
                    "summary": self._data["changelog"],
                    "file": f"runs/{run_tag}.json",
                }
            )
            _save_index(index)

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def run_tag(self) -> str:
        return self._data["run_tag"]

    def __repr__(self) -> str:
        return f"RunTracker(run_tag={self.run_tag!r})"


# ------------------------------------------------------------------ #
#  Helper: collect output artifacts from dependency runs              #
# ------------------------------------------------------------------ #

def _resolve_input_artifacts(depends_on: list[str]) -> list[dict]:
    """Return output_artifacts from each run listed in ``depends_on``."""
    inputs = []
    for tag in depends_on:
        dep = _load_run(tag)
        if dep is None:
            print(f"[RunTracker] Warning: dependency run '{tag}' not found in runs/")
            continue
        for kind, paths in dep.get("output_artifacts", {}).items():
            for path in paths:
                inputs.append({"run_tag": tag, "kind": kind, "path": path})
    return inputs
