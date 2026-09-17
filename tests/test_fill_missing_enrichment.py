from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fill_missing_enrichment",
    ROOT / "scripts" / "fill_missing_enrichment.py",
)
assert SPEC and SPEC.loader
fill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fill
SPEC.loader.exec_module(fill)

from src.modules.enricher import Enricher

VALID_ENRICHMENT = {
    "component": "DataNode",
    "log_level": "WARN",
    "operation": "serve",
    "fields": [],
    "explicit_conditions": [],
    "metadata_confidence": "medium",
    "component_role": "storage worker",
    "event_semantics": "Got an exception while serving a client.",
    "diagnostic_role": "warning_or_error",
    "failure_signals": [],
    "sequence_context": [],
    "dataset_label_caveat": (
        "HDFS-v1 labels apply to complete traces grouped by block ID, not to an "
        "individual log event or template."
    ),
    "embedding_text": "A DataNode reports an exception while serving.",
    "unsupported_inferences": [],
}


def _template_record(**overrides: object) -> dict:
    record = {
        "cluster_id": 30,
        "template": "WARN dfs.DataNode$DataXceiver: <IP>:Got exception while serv",
        "count": 12,
        "examples": ["WARN dfs.DataNode$DataXceiver: 10.0.0.1:Got exception while serv"],
    }
    record.update(overrides)
    return record


def test_enrichment_problem_detects_missing_and_invalid_caveat() -> None:
    assert fill.enrichment_problem(_template_record()) == "missing enriched_large"
    invalid = {**VALID_ENRICHMENT}
    del invalid["dataset_label_caveat"]
    problem = fill.enrichment_problem(_template_record(enriched_large=invalid))
    assert problem is not None
    assert "dataset_label_caveat" in problem
    assert fill.enrichment_problem(_template_record(enriched_large=VALID_ENRICHMENT)) is None


def test_export_writes_original_enricher_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    templates_path = tmp_path / "templates.json"
    templates_path.write_text(json.dumps([_template_record()]))
    output_dir = tmp_path / "manual"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fill_missing_enrichment.py",
            "export",
            "--dataset",
            "hdfs",
            "--templates",
            str(templates_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert fill.main() == 0
    prompt = (output_dir / "cluster_30" / "prompt.txt").read_text()
    assert "EVENTS ARE NOT LABELS" in prompt
    assert "dataset_label_caveat" in prompt
    assert "WARN dfs.DataNode$DataXceiver" in prompt
    assert "HDFS-v1 labels apply to complete traces" in prompt


def test_apply_writes_validated_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    templates_path = tmp_path / "templates.json"
    templates_path.write_text(json.dumps([_template_record()]))
    response_dir = tmp_path / "manual" / "cluster_30"
    response_dir.mkdir(parents=True)
    (response_dir / "response.json").write_text(json.dumps(VALID_ENRICHMENT))
    monkeypatch.setattr(
        "sys.argv",
        [
            "fill_missing_enrichment.py",
            "apply",
            "--dataset",
            "hdfs",
            "--templates",
            str(templates_path),
            "--responses-dir",
            str(tmp_path / "manual"),
        ],
    )
    assert fill.main() == 0
    saved = json.loads(templates_path.read_text())
    assert saved[0]["enriched_large"]["dataset_label_caveat"].startswith("HDFS-v1")
    assert fill.enrichment_problem(saved[0]) is None


class _FakeEnricher:
    llm = object()


def test_fill_record_with_llm_uses_original_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fill,
        "invoke_enricher_prompt",
        lambda enricher, context: json.dumps(VALID_ENRICHMENT),
    )
    record = _template_record()
    parsed = fill.fill_record_with_llm(record, "hdfs", _FakeEnricher())
    assert parsed.component == "DataNode"
    assert fill.enrichment_problem(record) is None


def test_fill_record_with_llm_injects_missing_dataset_caveat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {key: value for key, value in VALID_ENRICHMENT.items() if key != "dataset_label_caveat"}
    monkeypatch.setattr(
        fill,
        "invoke_enricher_prompt",
        lambda enricher, context: json.dumps(payload),
    )
    record = _template_record()
    fill.fill_record_with_llm(record, "hdfs", _FakeEnricher())
    assert record["enriched_large"]["dataset_label_caveat"]
    assert fill.enrichment_problem(record) is None


def test_enricher_parse_fills_omitted_dataset_label_caveat() -> None:
    payload = {key: value for key, value in VALID_ENRICHMENT.items() if key != "dataset_label_caveat"}
    parsed = Enricher._parse_response(json.dumps(payload))
    assert parsed.dataset_label_caveat
    assert "omitted by the model" in " ".join(parsed.unsupported_inferences)
