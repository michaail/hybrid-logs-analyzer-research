#!/usr/bin/env python3
"""Fill templates that Stage 2 skipped, using the original enricher prompt.

``fill`` sends each missing template through ``TEMPLATE_PROMPT`` and the
configured Azure deployment (``AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO``),
then writes ``enriched_large`` after the same parse path as Stage 2.

If the model omits ``dataset_label_caveat``, the script inserts the dataset
context already supplied in that prompt and re-validates. Use ``fill --paste``
only when you want to type JSON by hand.

Example::

    python scripts/fill_missing_enrichment.py list --dataset hdfs --workspace-root .

    python scripts/fill_missing_enrichment.py fill --dataset hdfs --workspace-root .

    python scripts/fill_missing_enrichment.py fill --dataset hdfs --workspace-root . --cluster-id 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.enricher import Enricher
from src.modules.enricher.prompts import (
    BGL_DATASET_CONTEXT,
    HDFS_DATASET_CONTEXT,
    TEMPLATE_PROMPT,
)
from src.modules.enricher.schemas import EnrichedTemplate, TemplateContext
from src.modules.enrichment import load_enriched_templates

ENRICHED_FIELD = "enriched_large"
DEEPSEEK_DEPLOYMENT_ENV = "AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO"
MappingLike = dict[str, Any]


def _dataset_context(dataset: str) -> str:
    if dataset == "bgl":
        return BGL_DATASET_CONTEXT
    if dataset == "hdfs":
        return HDFS_DATASET_CONTEXT
    raise ValueError(f"Unsupported dataset {dataset!r}")


def build_template_context(record: MappingLike, dataset: str) -> TemplateContext:
    context = TemplateContext.from_template_record(
        record,
        candidate_relations=record.get("candidate_relations", []),
        retrieved_docs=record.get("retrieved_docs", []),
    )
    context.dataset_context = _dataset_context(dataset)
    return context


def render_enricher_prompt(context: TemplateContext) -> str:
    """Return the exact chat transcript TEMPLATE_PROMPT would send."""
    messages = TEMPLATE_PROMPT.format_messages(
        context_json=json.dumps(
            context.model_dump(exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
    )
    blocks = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        blocks.append(f"----- {role} -----\n{message.content.strip()}")
    return "\n\n".join(blocks) + "\n"


def enrichment_problem(record: MappingLike) -> str | None:
    """Return why ``enriched_large`` is unusable, or None if it validates."""
    if ENRICHED_FIELD not in record:
        return "missing enriched_large"
    value: Any = record[ENRICHED_FIELD]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return f"enriched_large is not valid JSON: {exc}"
    if not isinstance(value, dict):
        return f"enriched_large has type {type(value).__name__}, expected object"
    try:
        EnrichedTemplate.model_validate(value)
    except Exception as exc:
        return str(exc)
    return None


def missing_records(templates: list[MappingLike]) -> list[MappingLike]:
    return [record for record in templates if enrichment_problem(record)]


def find_stage2_templates(workspace: Path, dataset: str) -> Path:
    root = workspace / "artifacts" / "cache" / dataset / "stage2_enrich"
    if not root.exists():
        raise FileNotFoundError(
            f"No Stage 2 cache at {root}. Pass --templates, or run prepare first."
        )
    candidates = [
        path.parent / "templates.json"
        for path in root.glob("*/_SUCCESS.json")
        if (path.parent / "templates.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No templates.json under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_templates_path(args: argparse.Namespace) -> Path:
    if args.templates:
        path = Path(args.templates).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    workspace = Path(args.workspace_root).expanduser().resolve()
    return find_stage2_templates(workspace, args.dataset)


def load_templates(path: Path) -> list[MappingLike]:
    templates, _, _ = load_enriched_templates(path, preferred_size="large")
    if not isinstance(templates, list):
        raise ValueError(f"{path} must contain a JSON list of templates")
    return templates


def save_templates(path: Path, templates: list[MappingLike]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(templates, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_enrichment_text(raw: str) -> EnrichedTemplate:
    return Enricher._parse_response(raw)


def configured_enricher() -> Enricher:
    deployment = os.getenv(DEEPSEEK_DEPLOYMENT_ENV)
    missing = [
        name
        for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", DEEPSEEK_DEPLOYMENT_ENV)
        if not os.getenv(name)
    ]
    if missing:
        raise EnvironmentError(
            "Missing Azure settings: "
            + ", ".join(missing)
            + ". They are loaded from the environment / .env like Stage 2."
        )
    return Enricher(deployment)


def invoke_enricher_prompt(enricher: Enricher, context: TemplateContext) -> Any:
    chain = TEMPLATE_PROMPT | enricher.llm
    return chain.invoke(
        {
            "context_json": json.dumps(
                context.model_dump(exclude_none=True),
                ensure_ascii=False,
                indent=2,
            )
        }
    )


def parse_response_with_dataset_caveat(response: Any, dataset: str) -> EnrichedTemplate:
    """Re-parse after filling dataset_label_caveat from the prompt's dataset context."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        raise TypeError(f"Expected text JSON response, got {type(content).__name__}")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("The LLM response must be a JSON object.")
    payload.setdefault("dataset_label_caveat", _dataset_context(dataset))
    return Enricher._parse_response(json.dumps(payload))


def fill_record_with_llm(
    record: MappingLike,
    dataset: str,
    enricher: Enricher,
) -> EnrichedTemplate:
    """Run the original enricher prompt against the configured LLM."""
    context = build_template_context(record, dataset)
    response = invoke_enricher_prompt(enricher, context)
    try:
        parsed = Enricher._parse_response(response)
    except Exception:
        parsed = parse_response_with_dataset_caveat(response, dataset)
    record[ENRICHED_FIELD] = parsed.model_dump(mode="json")
    problem = enrichment_problem(record)
    if problem:
        raise ValueError(problem)
    return parsed


def select_records(
    templates: list[MappingLike], cluster_id: str | None
) -> list[MappingLike]:
    missing = missing_records(templates)
    if cluster_id is None:
        return missing
    wanted = str(cluster_id)
    match = [
        record
        for record in templates
        if str(record.get("cluster_id")) == wanted
    ]
    if not match:
        raise KeyError(f"No template with cluster_id={wanted}")
    record = match[0]
    if enrichment_problem(record) is None:
        raise ValueError(f"cluster_id={wanted} already has valid {ENRICHED_FIELD}")
    return [record]


def cluster_dir(output_dir: Path, cluster_id: Any) -> Path:
    return output_dir / f"cluster_{cluster_id}"


def cmd_list(args: argparse.Namespace) -> int:
    path = resolve_templates_path(args)
    templates = load_templates(path)
    missing = missing_records(templates)
    print(f"templates : {path}")
    print(f"total     : {len(templates)}")
    print(f"missing   : {len(missing)}")
    for record in missing:
        problem = enrichment_problem(record)
        preview = str(record.get("template", ""))[:80]
        print(f"  cluster_id={record.get('cluster_id')}: {problem}")
        print(f"    {preview}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    path = resolve_templates_path(args)
    templates = load_templates(path)
    records = select_records(templates, args.cluster_id)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        Path("enrichment_manual") / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for record in records:
        context = build_template_context(record, args.dataset)
        target = cluster_dir(output_dir, record["cluster_id"])
        target.mkdir(parents=True, exist_ok=True)
        prompt_path = target / "prompt.txt"
        context_path = target / "context.json"
        response_path = target / "response.json"
        prompt_path.write_text(render_enricher_prompt(context))
        context_path.write_text(
            json.dumps(context.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
            + "\n"
        )
        if not response_path.exists():
            response_path.write_text(
                "{\n  \"_paste_the_model_json_here\": true\n}\n"
            )
        index.append(
            {
                "cluster_id": record["cluster_id"],
                "template": record.get("template"),
                "problem": enrichment_problem(record),
                "prompt": str(prompt_path),
                "response": str(response_path),
            }
        )
        print(f"wrote {prompt_path}")
    (output_dir / "missing_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Exported {len(index)} prompt(s) under {output_dir}")
    print("Prefer `fill` to call the configured LLM. These files are a manual fallback.")
    return 0


def _load_response_file(path: Path) -> str:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty response file: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and payload.get("_paste_the_model_json_here"):
        raise ValueError(f"Placeholder not replaced: {path}")
    return text


def apply_record(record: MappingLike, raw_json: str) -> EnrichedTemplate:
    parsed = parse_enrichment_text(raw_json)
    record[ENRICHED_FIELD] = parsed.model_dump(mode="json")
    return parsed


def cmd_apply(args: argparse.Namespace) -> int:
    path = resolve_templates_path(args)
    templates = load_templates(path)
    by_id = {str(record.get("cluster_id")): record for record in templates}
    responses_dir = Path(args.responses_dir).expanduser().resolve()
    files = sorted(responses_dir.glob("cluster_*/response.json"))
    if args.cluster_id:
        files = [cluster_dir(responses_dir, args.cluster_id) / "response.json"]
    if not files:
        raise FileNotFoundError(f"No cluster_*/response.json under {responses_dir}")
    updated = 0
    for response_path in files:
        if not response_path.exists():
            print(f"skip missing {response_path}")
            continue
        cluster_id = response_path.parent.name.removeprefix("cluster_")
        record = by_id.get(cluster_id)
        if record is None:
            print(f"skip unknown cluster_id={cluster_id}")
            continue
        try:
            apply_record(record, _load_response_file(response_path))
        except Exception as exc:
            print(f"FAILED cluster_id={cluster_id}: {exc}")
            continue
        updated += 1
        print(f"applied cluster_id={cluster_id}")
    if updated:
        save_templates(path, templates)
        print(f"Wrote {updated} enrichment(s) to {path}")
    else:
        print("No templates updated.")
        return 1
    return 0


def _read_pasted_json() -> str:
    print("Paste the JSON object, then a line containing only END")
    lines: list[str] = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        lines.append(line)
    return "".join(lines).strip()


def cmd_fill(args: argparse.Namespace) -> int:
    path = resolve_templates_path(args)
    templates = load_templates(path)
    records = select_records(templates, args.cluster_id)
    if not records:
        print("Nothing to fill.")
        return 0
    if args.paste:
        return _cmd_fill_paste(path, templates, records, args.dataset)
    return _cmd_fill_llm(path, templates, records, args.dataset)


def _cmd_fill_llm(
    path: Path,
    templates: list[MappingLike],
    records: list[MappingLike],
    dataset: str,
) -> int:
    enricher = configured_enricher()
    print(f"LLM        : {os.getenv(DEEPSEEK_DEPLOYMENT_ENV)}")
    print(f"templates  : {path}")
    print(f"to fill    : {len(records)}")
    updated = 0
    failed = 0
    for record in records:
        cluster_id = record.get("cluster_id")
        preview = str(record.get("template", ""))[:80]
        print(f"[LLM] cluster_id={cluster_id}  {preview}")
        try:
            fill_record_with_llm(record, dataset, enricher)
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")
            continue
        updated += 1
        save_templates(path, templates)
        print(f"  saved {ENRICHED_FIELD}")
    print(f"Updated {updated}/{len(records)} template(s); failed {failed}.")
    return 0 if failed == 0 else 1


def _cmd_fill_paste(
    path: Path,
    templates: list[MappingLike],
    records: list[MappingLike],
    dataset: str,
) -> int:
    updated = 0
    for record in records:
        cluster_id = record.get("cluster_id")
        print("=" * 72)
        print(f"cluster_id={cluster_id}")
        print(f"problem   ={enrichment_problem(record)}")
        print(f"template  ={record.get('template')}")
        print()
        print(render_enricher_prompt(build_template_context(record, dataset)))
        raw = _read_pasted_json()
        if not raw:
            print("empty paste; skipped")
            continue
        try:
            apply_record(record, raw)
        except Exception as exc:
            print(f"Validation failed: {exc}")
            continue
        updated += 1
        save_templates(path, templates)
        print(f"Saved {ENRICHED_FIELD} for cluster_id={cluster_id} → {path}")
    print(f"Updated {updated}/{len(records)} template(s).")
    return 0 if updated or not records else 1


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset", choices=("hdfs", "bgl"), default="hdfs")
    shared.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.environ.get("PIPELINE_WORKSPACE_ROOT", ".")),
    )
    shared.add_argument(
        "--templates",
        type=Path,
        help="templates.json to edit. Default: newest Stage 2 cache for --dataset.",
    )
    shared.add_argument("--cluster-id", help="Restrict to one cluster_id.")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", parents=[shared], help="Show templates missing a valid enriched_large.")
    export = sub.add_parser("export", parents=[shared], help="Write original enricher prompts to a folder.")
    export.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: enrichment_manual/<dataset>/",
    )
    apply = sub.add_parser("apply", parents=[shared], help="Read response.json files and write templates.json.")
    apply.add_argument("--responses-dir", type=Path, required=True)
    fill = sub.add_parser(
        "fill",
        parents=[shared],
        help="Call the configured LLM with the original enricher prompt.",
    )
    fill.add_argument(
        "--paste",
        action="store_true",
        help="Paste JSON in the terminal instead of calling Azure.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "list": cmd_list,
        "export": cmd_export,
        "apply": cmd_apply,
        "fill": cmd_fill,
    }
    try:
        return commands[args.command](args)
    except (FileNotFoundError, KeyError, ValueError, EnvironmentError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
