"""Stage 2 — LLM template semantic enrichment.

Wraps the existing :class:`src.enricher.Enricher` and adds
batch-level helpers consumed by :mod:`run_ablation`.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bumped when the enricher system prompt changes. Stage-2 cache keys omit git SHA,
# so this constant is what forces a re-enrich instead of reusing bland templates.
ENRICHMENT_PROMPT_VERSION = "bgl_extended_v1"


class IncompleteEnrichmentError(ValueError):
    """Raised when a known template lacks a validated frozen LLM response."""


def _canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-serialisable provenance."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enrichment_provenance(
    templates_data: list[dict],
    *,
    dataset: str,
    enabled: bool,
    deployment: str | None = None,
) -> dict[str, Any]:
    """Describe frozen LLM inputs and outputs without storing credentials.

    The context digest covers exactly the template records passed to the
    enricher, including sanitized examples. The response digest covers the
    validated `enriched_large` records. Both are checked before an LLM graph
    can be built from a cached stage-2 result.
    """
    prompt_dir = Path(__file__).parent / "enricher" / "prompts"
    prompt_sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(prompt_dir.glob("*.py"))
    }
    contexts = [
        {
            "cluster_id": int(entry.get("cluster_id", index)),
            "template": entry.get("template", ""),
            "examples": entry.get("examples", []),
            "candidate_relations": entry.get("candidate_relations", []),
            "retrieved_docs": entry.get("retrieved_docs", []),
        }
        for index, entry in enumerate(templates_data)
        if int(entry.get("cluster_id", index)) >= 0
    ]
    responses = [
        {
            "cluster_id": int(entry.get("cluster_id", index)),
            "enriched_large": entry.get("enriched_large"),
        }
        for index, entry in enumerate(templates_data)
        if int(entry.get("cluster_id", index)) >= 0
    ]
    return {
        "schema_version": 1,
        "dataset": dataset.lower(),
        "enabled": bool(enabled),
        "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "prompt_sources_sha256": prompt_sources,
        "prompt_sha256": _canonical_digest(prompt_sources),
        "deployment": deployment if enabled else None,
        "decoding": {"temperature": 0, "max_tokens": None, "max_retries": 2},
        "n_known_templates": len(contexts),
        "context_sha256": _canonical_digest(contexts),
        "response_sha256": _canonical_digest(responses) if enabled else None,
    }


def require_valid_enrichment_provenance(
    templates_data: list[dict],
    provenance: dict[str, Any],
    *,
    dataset: str,
) -> None:
    """Fail closed if frozen enrichment provenance is absent or inconsistent."""
    if not provenance.get("enabled"):
        raise IncompleteEnrichmentError("LLM enrichment provenance is not marked enabled.")
    if provenance.get("dataset") != dataset.lower():
        raise IncompleteEnrichmentError("LLM enrichment provenance dataset does not match.")
    if not provenance.get("deployment"):
        raise IncompleteEnrichmentError("LLM enrichment provenance lacks a deployment identifier.")
    require_complete_enrichment(templates_data)
    expected = enrichment_provenance(
        templates_data,
        dataset=dataset,
        enabled=True,
        deployment=str(provenance["deployment"]),
    )
    for key in ("prompt_version", "prompt_sha256", "context_sha256", "response_sha256"):
        if provenance.get(key) != expected[key]:
            raise IncompleteEnrichmentError(f"LLM enrichment provenance mismatch for {key}.")


def require_complete_enrichment(
    templates_data: list[dict],
    *,
    field: str = "enriched_large",
) -> None:
    """Fail closed when a non-OOV template lacks usable enrichment data."""
    missing = [
        int(entry.get("cluster_id", index))
        for index, entry in enumerate(templates_data)
        if int(entry.get("cluster_id", 0)) >= 0
        and not isinstance(entry.get(field), dict)
    ]
    if missing:
        raise IncompleteEnrichmentError(
            f"Missing validated {field} data for {len(missing)} known template(s): "
            f"cluster_ids={missing[:20]}"
        )


def enrich_templates(
    templates_data: list[dict],
    dataset: str,
    model_size: str = "large",
    enrichment_profile: str = "grounded",
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
        if int(entry.get("cluster_id", 0)) < 0:
            logger.info("Skipping OOV template cluster_id=%s", entry.get("cluster_id"))
            continue
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
                    "anomalous event. Enrichment profile: "
                    f"{enrichment_profile}."
                )

            result = enricher.enrich_template(context)
            entry[field] = result.model_dump(mode="json")
            logger.debug("Enriched template %d/%d", i + 1, len(templates_data))
        except Exception as exc:
            logger.warning(
                "Failed to enrich template %d (%r): %s", i + 1, template[:60], exc
            )

    require_complete_enrichment(templates_data, field=field)

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
