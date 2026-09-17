"""Stage 2 — LLM template semantic enrichment.

Wraps the existing :class:`src.enricher.Enricher` and adds
batch-level helpers consumed by :mod:`run_ablation`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def enrich_templates(
    templates_data: list[dict],
    dataset: str,
    model_size: str = "large",
) -> list[dict]:
    """Enrich a list of template dicts with LLM semantic annotations.

    Parameters
    ----------
    templates_data:
        List of dicts with at least a ``"template"`` key (output of the parser
        stage).  Adds ``"enriched_large"`` (Deepseek v4 Pro) in-place.
    dataset:
        ``"hdfs"`` or ``"bgl"`` — selects the enrichment prompt.
    model_size:
        Kept for cache/identity compatibility. Ablation enrichment always uses
        Azure ``AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO`` and stores the result
        in ``enriched_large``. ``small`` / ``both`` are not valid arms.

    Returns
    -------
    list[dict]
        The enriched templates list (same object, modified in-place).
    """
    from src.modules.enricher import Enricher, TemplateContext  # optional dependency

    if model_size in {"small", "both"}:
        raise ValueError(
            "Ablation enrichment uses Deepseek v4 Pro only; "
            f"model_size={model_size!r} is not supported."
        )
    deployment_env = "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO"
    deployment = os.getenv(deployment_env)
    if not deployment:
        raise EnvironmentError(
            f"Environment variable {deployment_env!r} is not set. "
            "Cannot run LLM enrichment."
        )

    enricher = Enricher(deployment)
    field = "enriched_large"

    for i, entry in enumerate(templates_data):
        template = entry["template"]
        try:
            context = TemplateContext.from_template_record(
                entry,
                candidate_relations=entry.get("candidate_relations", []),
                retrieved_docs=entry.get("retrieved_docs", []),
            )
            if dataset.lower() == "hdfs":
                context.dataset_context = (
                    "HDFS-v1 labels apply to complete traces grouped by block ID, not "
                    "to an individual log event or template."
                )
            elif dataset.lower() == "bgl":
                context.dataset_context = (
                    "BGL source log messages carry event-level labels. When transformed "
                    "into time windows, a window is anomalous when it contains an "
                    "anomalous event."
                )

            result = enricher.enrich_template(context)
            entry[field] = result.model_dump(mode="json")
            logger.debug("Enriched template %d/%d", i + 1, len(templates_data))
        except Exception as exc:
            logger.warning(
                "Failed to enrich template %d (%r): %s", i + 1, template[:60], exc
            )

    return templates_data


def load_enriched_templates(
    path: str | Path,
    *,
    preferred_size: str | None = None,
) -> tuple[list[dict], dict[int, str], dict[int, Any]]:
    """Load an enriched (or plain) templates JSON file and return look-ups.

    Gracefully falls back when no enrichment fields are present, so callers
    can use this function regardless of whether Stage 2 ran.

    Parameters
    ----------
    preferred_size:
        ``"large"`` or ``"small"`` selects ``enriched_*`` first when both
        fields are stored in the same JSON (campaign prepare writes both).

    Returns
    -------
    templates_data : list[dict]
        Raw list as stored on disk.
    cluster_to_template : dict[int, str]
        ``cluster_id → template string``.
    cluster_to_enriched : dict[int, Any]
        ``cluster_id → parsed enrichment dict``.  Empty if enrichment was
        disabled or the file contains only raw templates.
    """
    path = Path(path)
    with open(path) as fh:
        templates_data = json.load(fh)

    cluster_to_template: dict[int, str] = {
        t["cluster_id"]: t["template"] for t in templates_data
    }
    fields: list[str] = []
    if preferred_size in {"large", "small"}:
        fields.append(f"enriched_{preferred_size}")
    for name in ("enriched_large", "enriched_small"):
        if name not in fields:
            fields.append(name)
    cluster_to_enriched: dict[int, Any] = {}
    for t in templates_data:
        for field in fields:
            if field not in t:
                continue
            value = t[field]
            if isinstance(value, dict):
                cluster_to_enriched[t["cluster_id"]] = value
            elif isinstance(value, str):
                try:
                    cluster_to_enriched[t["cluster_id"]] = json.loads(value)
                except json.JSONDecodeError:
                    pass
            break

    return templates_data, cluster_to_template, cluster_to_enriched
