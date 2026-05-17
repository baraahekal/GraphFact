from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_metric.data_io import (
    SourceSentence,
    SummaryExample,
    clean_text,
    load_config,
    parse_plain_source_sentences,
    sentence_split,
)
from graph_metric.metric import GraphMetric as CoreGraphMetric


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "graph_metric.yaml"


class GraphMetric:
    """Public source-document factuality metric API.

    This wrapper keeps dataset adapters out of normal usage. Users provide a source
    document and a generated summary; the core metric returns the score, graph
    evidence, and diagnostics.
    """

    def __init__(self, config_path: str | Path | None = None, config: dict[str, Any] | None = None):
        if config is None:
            config = load_config(config_path or DEFAULT_CONFIG_PATH)
        self.config = config
        self._metric = CoreGraphMetric(config)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "GraphMetric":
        return cls(config_path=config_path)

    def score(
        self,
        source_doc: str,
        summary: str,
        *,
        document_id: str = "document",
        summary_id: str = "summary",
        sections: list[dict[str, str]] | dict[str, str] | None = None,
        abstract_text: str = "",
    ) -> dict[str, Any]:
        example = build_example(
            source_doc=source_doc,
            summary=summary,
            document_id=document_id,
            summary_id=summary_id,
            sections=sections,
            abstract_text=abstract_text,
        )
        claim_rows, graph_rows, sentence_rows, summary_rows = self._metric.score_examples([example])
        summary_row = summary_rows[0] if summary_rows else {}
        return {
            "metric": "GraphLongDocFACTScore",
            "document_id": document_id,
            "summary_id": summary_id,
            "summary_score": float(summary_row.get("graph_summary_score", 0.0)),
            "num_summary_sentences": int(summary_row.get("num_summary_sentences", 0)),
            "num_claims": int(summary_row.get("num_claims", len(claim_rows))),
            "summary": summary_row,
            "sentences": sentence_rows,
            "claims": claim_rows,
            "graphs": graph_rows,
            "config": public_config(self.config),
        }


def build_example(
    *,
    source_doc: str,
    summary: str,
    document_id: str,
    summary_id: str,
    sections: list[dict[str, str]] | dict[str, str] | None = None,
    abstract_text: str = "",
) -> SummaryExample:
    source_sentences = source_sentences_from_sections(document_id, sections) if sections else parse_plain_source_sentences(document_id, source_doc)
    summary_sentences = sentence_split(summary)
    if not source_sentences:
        raise ValueError("source_doc or sections must contain at least one sentence")
    if not summary_sentences:
        raise ValueError("summary must contain at least one sentence")
    return SummaryExample(
        document_id=document_id,
        summary_model=summary_id,
        summary_model_name=summary_id,
        summary_text=clean_text(summary),
        summary_sentences=summary_sentences,
        human_summary_score=0.0,
        source_sentences=source_sentences,
        abstract_text=clean_text(abstract_text),
    )


def source_sentences_from_sections(
    document_id: str,
    sections: list[dict[str, str]] | dict[str, str],
) -> list[SourceSentence]:
    if isinstance(sections, dict):
        normalized_sections = [{"name": name, "text": text} for name, text in sections.items()]
    else:
        normalized_sections = sections

    source_sentences: list[SourceSentence] = []
    global_index = 0
    for section_index, section in enumerate(normalized_sections):
        section_name = clean_text(section.get("name", f"Section {section_index + 1}"))
        section_text = clean_text(section.get("text", ""))
        section_id = f"{document_id}_sec{section_index:03d}"
        for section_sentence_index, sentence in enumerate(sentence_split(section_text)):
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=section_name,
                    section_index=section_index,
                    sentence_index=global_index,
                    section_sentence_index=section_sentence_index,
                    text=sentence,
                )
            )
            global_index += 1
    return source_sentences


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_mode": config.get("claim_extraction", {}).get("mode", "sentence"),
        "retrieval": {
            "mode": config.get("retrieval", {}).get("mode"),
            "embedding_model": config.get("retrieval", {}).get("embedding_model"),
            "top_k": config.get("retrieval", {}).get("top_k"),
            "neighbor_window": config.get("retrieval", {}).get("neighbor_window"),
        },
        "scoring": {
            "use_nli": config.get("scoring", {}).get("use_nli"),
            "nli_model": config.get("scoring", {}).get("nli_model"),
            "use_bartscore": config.get("scoring", {}).get("use_bartscore"),
            "sentence_aggregation": config.get("scoring", {}).get("sentence_aggregation"),
            "claim_aggregation": config.get("scoring", {}).get("claim_aggregation"),
            "summary_aggregation": config.get("scoring", {}).get("summary_aggregation"),
        },
    }
