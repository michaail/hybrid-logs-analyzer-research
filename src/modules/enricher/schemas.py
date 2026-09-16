from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator


MetadataConfidence = Literal["high", "medium", "low", "unknown"]
MetadataSource = Literal[
    "template",
    "examples",
    "corpus_relation",
    "documentation",
    "unknown",
]
DiagnosticRole = Literal[
    "informational",
    "lifecycle_transition",
    "warning_or_error",
    "context_dependent",
    "unknown",
]
RelationType = Literal[
    "commonly_precedes",
    "commonly_follows",
    "co_occurs_in_trace",
    "same_component_lifecycle",
    "potential_rca_context",
]


class TemplateContext(BaseModel):
    """Format-agnostic evidence supplied to one enrichment call per template."""

    template_id: str
    template: str
    occurrence_count: int = Field(ge=0)
    examples: list[str] = Field(default_factory=list, max_length=5)
    candidate_relations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    dataset_context: str | None = None

    @field_validator("examples")
    @classmethod
    def remove_empty_examples(cls, examples: list[str]) -> list[str]:
        return [example.strip() for example in examples if example.strip()]

    @classmethod
    def from_template_record(
        cls,
        record: Mapping[str, Any],
        *,
        candidate_relations: Sequence[Mapping[str, Any]] | None = None,
        retrieved_docs: Sequence[Mapping[str, Any]] | None = None,
        dataset_context: str | None = None,
    ) -> "TemplateContext":
        """Create context from a mined-template record without format-specific rules."""
        examples = record.get("examples", [])
        return cls(
            template_id=str(record["cluster_id"]),
            template=str(record["template"]),
            occurrence_count=int(record.get("count", 0)),
            examples=[str(example) for example in examples[:5]],
            candidate_relations=[dict(item) for item in candidate_relations or ()],
            retrieved_docs=[dict(item) for item in retrieved_docs or ()],
            dataset_context=dataset_context,
        )


class TemplateField(BaseModel):
    placeholder: str
    semantic_role: str
    source: MetadataSource = "unknown"
    confidence: MetadataConfidence = "unknown"


class TemplateRelation(BaseModel):
    template_id: str
    relation: RelationType
    support: str
    source: MetadataSource = "corpus_relation"


class FailureSignal(BaseModel):
    name: str
    manifestation: str
    trigger_scope: Literal[
        "explicit_in_template",
        "requires_sequence_context",
        "requires_external_metric",
    ]
    source: MetadataSource = "unknown"
    confidence: MetadataConfidence = "unknown"


class EnrichedTemplate(BaseModel):
    """Observed template metadata and semantic enrichment from one LLM response."""

    component: str
    log_level: str
    operation: str
    fields: list[TemplateField] = Field(default_factory=list)
    explicit_conditions: list[str] = Field(default_factory=list)
    metadata_confidence: MetadataConfidence
    component_role: str
    event_semantics: str
    diagnostic_role: DiagnosticRole
    failure_signals: list[FailureSignal] = Field(default_factory=list)
    sequence_context: list[TemplateRelation] = Field(default_factory=list)
    dataset_label_caveat: str
    embedding_text: str
    unsupported_inferences: list[str] = Field(default_factory=list)
