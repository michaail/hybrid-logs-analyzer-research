from collections.abc import Mapping, Sequence
import json
from typing import Any

from .llm import LLM
from .prompts import BGL_DATASET_CONTEXT, HDFS_DATASET_CONTEXT, TEMPLATE_PROMPT
from .schemas import EnrichedTemplate, TemplateContext


class Enricher:
    def __init__(self, model: str | None = None) -> None:
        llm = LLM(model=model).get_llm()
        # Parse JSON manually because some Azure-hosted models wrap the object
        # in an extra key before Pydantic sees it.
        self.llm = llm

    def enrich_template(self, context: TemplateContext) -> EnrichedTemplate:
        """Infer metadata and semantic enrichment in one LLM call.

        ``TemplateContext`` contains only evidence produced by template mining
        and optional corpus-level observations. It does not encode a log-format
        parser or domain-specific field assumptions.
        """
        chain = TEMPLATE_PROMPT | self.llm
        response = chain.invoke(
            {
                "context_json": json.dumps(
                    context.model_dump(exclude_none=True),
                    ensure_ascii=False,
                    indent=2,
                )
            }
        )
        return self._parse_response(response)

    def enrich_corpus_hdfs(
        self,
        template: str,
        *,
        template_id: str = "unknown",
        examples: Sequence[str] | None = None,
        occurrence_count: int = 0,
        retrieved_docs: Sequence[Mapping[str, Any]] | None = None,
        candidate_relations: Sequence[dict[str, Any]] | None = None,
    ) -> EnrichedTemplate:
        """Backward-compatible HDFS wrapper around :meth:`enrich_template`."""
        return self.enrich_template(
            TemplateContext(
                template_id=template_id,
                template=template,
                occurrence_count=occurrence_count,
                examples=list(examples or []),
                retrieved_docs=[dict(item) for item in retrieved_docs or ()],
                candidate_relations=list(candidate_relations or []),
                dataset_context=HDFS_DATASET_CONTEXT,
            )
        )

    def enrich_corpus_bgl(
        self,
        template: str,
        *,
        template_id: str = "unknown",
        examples: Sequence[str] | None = None,
        occurrence_count: int = 0,
        retrieved_docs: Sequence[Mapping[str, Any]] | None = None,
        candidate_relations: Sequence[dict[str, Any]] | None = None,
    ) -> EnrichedTemplate:
        """Backward-compatible BGL wrapper around :meth:`enrich_template`."""
        return self.enrich_template(
            TemplateContext(
                template_id=template_id,
                template=template,
                occurrence_count=occurrence_count,
                examples=list(examples or []),
                retrieved_docs=[dict(item) for item in retrieved_docs or ()],
                candidate_relations=list(candidate_relations or []),
                dataset_context=BGL_DATASET_CONTEXT,
            )
        )

    @staticmethod
    def _parse_response(response: Any) -> EnrichedTemplate:
        """Parse and conservatively repair provider-specific JSON responses."""
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise TypeError(f"Expected text JSON response, got {type(content).__name__}")

        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()

        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("The LLM response must be a JSON object.")

        has_enrichment_wrapper = isinstance(payload.get("enrichment"), dict)
        payload = Enricher._flatten_sections(payload)
        payload = Enricher._normalise_root_aliases(payload)
        payload["failure_signals"], failure_signal_notes = (
            Enricher._normalise_failure_signals(payload.get("failure_signals", []))
        )
        payload["fields"] = Enricher._normalise_fields(payload.get("fields", []))
        payload["sequence_context"] = Enricher._normalise_relations(
            payload.get("sequence_context", [])
        )
        payload.setdefault("explicit_conditions", [])
        unsupported = payload.setdefault("unsupported_inferences", [])
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]
            payload["unsupported_inferences"] = unsupported
        unsupported.extend(failure_signal_notes)
        payload["metadata_confidence"], confidence_note = (
            Enricher._normalise_confidence(payload.get("metadata_confidence"))
        )
        if confidence_note:
            unsupported.append(confidence_note)
        payload["diagnostic_role"], role_note = Enricher._normalise_diagnostic_role(
            payload.get("diagnostic_role")
        )
        if role_note:
            unsupported.append(role_note)
        if has_enrichment_wrapper:
            unsupported.append("Provider returned a nested enrichment object; it was unwrapped.")

        return EnrichedTemplate.model_validate(payload)

    @staticmethod
    def _flatten_sections(payload: dict[str, Any]) -> dict[str, Any]:
        """Accept sectioned responses without making sectioning part of the API."""
        if not isinstance(payload.get("metadata"), dict) and not isinstance(
            payload.get("enrichment"), dict
        ):
            return payload

        result: dict[str, Any] = {}
        metadata = payload.get("metadata", {})
        enrichment = payload.get("enrichment", {})
        if isinstance(metadata, dict):
            result.update(metadata)
        if isinstance(enrichment, dict):
            result.update(enrichment)
        for key, value in payload.items():
            if key not in {"metadata", "enrichment"}:
                result[key] = value
        return result

    @staticmethod
    def _normalise_root_aliases(payload: dict[str, Any]) -> dict[str, Any]:
        """Repair common naming variants without inventing missing claims."""
        result = dict(payload)
        aliases = {
            "emitter_or_component": "component",
            "severity_or_priority": "log_level",
            "parameters": "fields",
            "conditions": "explicit_conditions",
            "confidence": "metadata_confidence",
        }
        for source, target in aliases.items():
            if target not in result and source in result:
                result[target] = result[source]
        return result

    @staticmethod
    def _normalise_fields(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        fields: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            field = dict(item)
            field.setdefault("placeholder", field.pop("name", "unknown"))
            field.setdefault(
                "semantic_role",
                field.pop("role", field.pop("description", "unknown")),
            )
            field.setdefault("source", field.pop("evidence", "unknown"))
            if field["source"] not in {
                "template",
                "examples",
                "corpus_relation",
                "documentation",
            }:
                field["source"] = "unknown"
            field["confidence"], _ = Enricher._normalise_confidence(
                field.get("confidence")
            )
            fields.append(field)
        return fields

    @staticmethod
    def _normalise_relations(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        relations: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            relation = dict(item)
            relation.setdefault("template_id", "unknown")
            relation.setdefault("relation", "potential_rca_context")
            relation.setdefault("support", "unknown")
            if relation["relation"] == "co-occurrence":
                relation["relation"] = "co_occurs_in_trace"
            relation.setdefault("source", "corpus_relation")
            if relation["source"] not in {
                "template",
                "examples",
                "corpus_relation",
                "documentation",
            }:
                relation["source"] = "unknown"
            relations.append(relation)
        return relations

    @staticmethod
    def _normalise_failure_signals(
        value: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Map common abbreviated provider output to the strict signal schema."""
        if value is None:
            return [], []
        if not isinstance(value, list):
            value = [value]

        signals: list[dict[str, Any]] = []
        notes: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            signal = dict(item)
            signal.setdefault("name", signal.pop("signal", "unknown"))
            signal.setdefault(
                "manifestation",
                signal.pop("description", signal.pop("observable_signal", "unknown")),
            )
            signal["trigger_scope"], scope_note = Enricher._normalise_trigger_scope(
                signal.get("trigger_scope")
            )
            if scope_note:
                notes.append(
                    f"Failure signal {signal['name']!r}: {scope_note}"
                )
            signal.setdefault("source", signal.pop("evidence", "unknown"))
            if signal["source"] not in {
                "template",
                "examples",
                "corpus_relation",
                "documentation",
            }:
                signal["source"] = "unknown"
            signal["confidence"], _ = Enricher._normalise_confidence(
                signal.get("confidence")
            )
            signals.append(signal)
        return signals, notes

    @staticmethod
    def _normalise_trigger_scope(value: Any) -> tuple[str, str | None]:
        """Translate clear scope variants without treating ambiguous signals as local."""
        valid = {
            "explicit_in_template",
            "requires_sequence_context",
            "requires_external_metric",
        }
        if isinstance(value, str):
            normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "single_block_operation": "explicit_in_template",
                "single_event": "explicit_in_template",
                "event_level": "explicit_in_template",
                "template": "explicit_in_template",
                "sequence": "requires_sequence_context",
                "trace": "requires_sequence_context",
                "lifecycle": "requires_sequence_context",
                "external_metric": "requires_external_metric",
                "metric": "requires_external_metric",
            }
            scope = aliases.get(normalised, normalised)
            if scope in valid:
                return scope, None

        return "requires_sequence_context", (
            f"Provider returned unsupported trigger_scope {value!r}; "
            "recorded as 'requires_sequence_context' to avoid treating the signal "
            "as independently diagnostic."
        )

    @staticmethod
    def _normalise_confidence(value: Any) -> tuple[str, str | None]:
        """Return the schema's confidence enum without overstating evidence."""
        valid = ("high", "medium", "low", "unknown")
        if isinstance(value, str):
            normalised = value.strip().lower().replace(" ", "_")
            if normalised in valid:
                return normalised, None
            for confidence in valid:
                if confidence in normalised:
                    return (
                        confidence,
                        f"Provider used non-canonical confidence value {value!r}; "
                        f"normalised to {confidence!r}.",
                    )
        elif isinstance(value, dict):
            reported = [
                item.strip().lower()
                for item in value.values()
                if isinstance(item, str) and item.strip().lower() in valid
            ]
            if reported:
                # A single root confidence must not be stronger than any
                # field-level confidence the provider supplied.
                confidence = max(reported, key=valid.index)
                return (
                    confidence,
                    "Provider returned per-field confidence instead of one "
                    f"metadata_confidence value; conservatively used {confidence!r}.",
                )

        return "unknown", (
            f"Provider returned unsupported confidence value {value!r}; "
            "recorded as 'unknown'."
        )

    @staticmethod
    def _normalise_diagnostic_role(value: Any) -> tuple[str, str | None]:
        """Map only clear diagnostic-role variants; preserve ambiguity as unknown."""
        valid = {
            "informational",
            "lifecycle_transition",
            "warning_or_error",
            "context_dependent",
            "unknown",
        }
        if isinstance(value, str):
            normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "info": "informational",
                "information": "informational",
                "lifecycle": "lifecycle_transition",
                "transition": "lifecycle_transition",
                "warning": "warning_or_error",
                "error": "warning_or_error",
                "warning/error": "warning_or_error",
                "warning_or_error": "warning_or_error",
                "contextual": "context_dependent",
            }
            role = aliases.get(normalised, normalised)
            if role in valid:
                return role, None

        return "unknown", (
            f"Provider returned non-categorical diagnostic_role {value!r}; "
            "recorded as 'unknown'."
        )
