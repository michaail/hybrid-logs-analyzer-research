from __future__ import annotations

import pytest

from src.modules.dataset import (
    NODE_EXTRA_DIM,
    SBERT_TEXT_EMBEDDING,
    SBERT_TEXT_GROUNDED,
    compose_sbert_text,
    embedding_block_dims,
    node_recon_indices,
    sbert_dim_from_meta,
)
from src.modules.enrichment import (
    IncompleteEnrichmentError,
    enrichment_provenance,
    require_complete_enrichment,
    require_valid_enrichment_provenance,
)
from src.modules.parser.bgl_parser import BGLParser


CID3_TEMPLATE = "RAS KERNEL <*> <*> <*>"
CID3_EXAMPLE = (
    "- 1117839085 2005.06.03 R20-M1-N5-C:J17-U01 2005-06-03-15.51.25.712950 "
    "R20-M1-N5-C:J17-U01 RAS KERNEL INFO generating core.304"
)
PANIC_TEMPLATE = "RAS KERNEL FATAL rts panic! - stopping execution"

ENRICHED_PANIC = {
    "log_level": "FATAL",
    "diagnostic_role": "warning_or_error",
    "operation": "rts panic",
    "event_semantics": "The RAS kernel stops execution after a runtime panic.",
    "embedding_text": (
        "RAS KERNEL FATAL rts panic: the kernel is stopping execution. "
        "The template reports an unrecoverable runtime stop."
    ),
    "failure_signals": [
        {
            "name": "kernel_panic",
            "manifestation": "rts panic stopping execution",
            "trigger_scope": "explicit_in_template",
        },
        {
            "name": "invented_reboot",
            "manifestation": "HEARTBEAT after reboot",
            "trigger_scope": "requires_sequence_context",
        },
    ],
}


def test_grounded_sbert_text_is_enrichment_only() -> None:
    text = compose_sbert_text(
        PANIC_TEMPLATE,
        ENRICHED_PANIC,
        mode=SBERT_TEXT_GROUNDED,
    )
    assert "rts panic" in text
    assert "FATAL" in text
    assert "kernel_panic" in text
    assert "HEARTBEAT" not in text
    assert CID3_TEMPLATE not in text
    assert "generating core" not in text


def test_grounded_sbert_text_ignores_drain_template_and_examples() -> None:
    text = compose_sbert_text(
        CID3_TEMPLATE,
        {
            "embedding_text": "Examples show the kernel generating a core file.",
            "log_level": "INFO",
            "diagnostic_role": "informational",
            "operation": "generating core",
        },
        mode=SBERT_TEXT_GROUNDED,
    )
    assert "generating a core file" in text
    assert "<*>" not in text
    assert CID3_EXAMPLE not in text


def test_grounded_without_enrichment_is_unknown() -> None:
    assert compose_sbert_text(CID3_TEMPLATE, None, mode=SBERT_TEXT_GROUNDED) == (
        "unknown log template"
    )


def test_embedding_text_mode_uses_llm_paragraph() -> None:
    text = compose_sbert_text(
        PANIC_TEMPLATE,
        ENRICHED_PANIC,
        mode=SBERT_TEXT_EMBEDDING,
    )
    assert text == ENRICHED_PANIC["embedding_text"]
    assert PANIC_TEMPLATE not in text


def test_embedding_text_mode_falls_back_to_template_when_llm_off() -> None:
    assert compose_sbert_text(PANIC_TEMPLATE, None, mode=SBERT_TEXT_EMBEDDING) == (
        PANIC_TEMPLATE
    )


def test_embedding_block_and_recon_slices() -> None:
    tfidf_dim, sbert_dim = embedding_block_dims(
        897, tfidf_enabled=True, sbert_enabled=True
    )
    assert (tfidf_dim, sbert_dim) == (513, 384)
    node_dim = 897 + NODE_EXTRA_DIM
    idx = node_recon_indices(node_dim, sbert_dim=384)
    assert list(idx[:3]) == [0, 1, 2]
    assert 513 not in set(idx)
    assert idx[-1] == node_dim - 1
    full = node_recon_indices(node_dim, sbert_dim=0)
    assert len(full) == node_dim


def test_sbert_dim_from_legacy_meta() -> None:
    assert (
        sbert_dim_from_meta(
            {
                "embed_dim": 897,
                "embedding_flags": {"tfidf_enabled": True, "sbert_enabled": True},
            }
        )
        == 384
    )
    assert sbert_dim_from_meta({"sbert_dim": 0, "embed_dim": 513}) == 0


def test_complete_enrichment_rejects_missing_known_template() -> None:
    templates = [
        {"cluster_id": 7, "template": "known"},
        {"cluster_id": -1, "template": "OOV"},
    ]
    with pytest.raises(IncompleteEnrichmentError, match="cluster_ids=\\[7\\]"):
        require_complete_enrichment(templates)


def test_complete_enrichment_allows_oov_and_validated_known_template() -> None:
    require_complete_enrichment(
        [
            {"cluster_id": 7, "template": "known", "enriched_large": {}},
            {"cluster_id": -1, "template": "OOV"},
        ]
    )


def test_enrichment_provenance_rejects_changed_frozen_context() -> None:
    templates = [
        {
            "cluster_id": 7,
            "template": "known",
            "examples": ["sanitized example"],
            "enriched_large": {"embedding_text": "frozen response"},
        },
        {"cluster_id": -1, "template": "OOV"},
    ]
    provenance = enrichment_provenance(
        templates, dataset="bgl", enabled=True, deployment="frozen-deployment"
    )
    require_valid_enrichment_provenance(templates, provenance, dataset="bgl")
    templates[0]["examples"] = ["different sanitized example"]
    with pytest.raises(IncompleteEnrichmentError, match="context_sha256"):
        require_valid_enrichment_provenance(templates, provenance, dataset="bgl")


def test_bgl_template_examples_exclude_inline_benchmark_label() -> None:
    raw = "KERNDTLB 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 payload"
    example = BGLParser._template_example(raw)
    assert example == "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 payload"
    assert "KERNDTLB" not in example
