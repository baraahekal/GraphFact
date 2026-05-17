from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph_metric.api import GraphMetric
from graph_metric.report import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a generated summary against a source document.")
    parser.add_argument("--source", required=True, help="Path to the full source document text file.")
    parser.add_argument("--summary", required=True, help="Path to the generated summary text file.")
    parser.add_argument("--config", default="configs/graph_metric.yaml", help="Metric config path.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--summary-id", default=None)
    parser.add_argument(
        "--sections-json",
        default=None,
        help="Optional JSON file containing [{\"name\": ..., \"text\": ...}] or {section_name: text}.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    summary_path = Path(args.summary)
    document_id = args.document_id or source_path.stem
    summary_id = args.summary_id or summary_path.stem
    sections = read_sections(args.sections_json) if args.sections_json else None

    metric = GraphMetric.from_config(args.config)
    result = metric.score(
        source_doc=source_path.read_text(encoding="utf-8"),
        summary=summary_path.read_text(encoding="utf-8"),
        document_id=document_id,
        summary_id=summary_id,
        sections=sections,
    )

    if args.output:
        write_json(args.output, result)
    print(json.dumps(short_result(result), indent=2, ensure_ascii=False))


def read_sections(path: str) -> list[dict[str, str]] | dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, (list, dict)):
        raise ValueError("--sections-json must contain a JSON list or object")
    return data


def short_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": result["metric"],
        "document_id": result["document_id"],
        "summary_id": result["summary_id"],
        "summary_score": result["summary_score"],
        "num_summary_sentences": result["num_summary_sentences"],
        "num_claims": result["num_claims"],
        "num_graphs": len(result["graphs"]),
    }


if __name__ == "__main__":
    main()
