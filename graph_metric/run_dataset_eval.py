from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_metric.data_io import load_config, load_examples
from graph_metric.metric import GraphMetric
from graph_metric.report import choose_best_binary_threshold, evaluate, write_csv, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphLongDocFACTScore on a supported benchmark dataset.")
    parser.add_argument("--metric-config", default="configs/graph_metric.yaml")
    parser.add_argument("--dataset", choices=["longsciverify", "longeval", "fenice", "screen_eval"], required=True)
    parser.add_argument("--subset", choices=["pubmed", "arxiv", "squality", "story_summeval", "dialogue"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test", "all"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluation-task", choices=["correlation", "binary"], default="correlation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold-on-validation", action="store_true")
    parser.add_argument("--validation-split", default="validation")
    args = parser.parse_args()

    config = load_config(args.metric_config)
    config["dataset"] = {
        "name": args.dataset,
        "subset": args.subset,
        "root": args.data_root,
    }
    if args.split:
        config["dataset"]["split"] = args.split
    config["outputs"] = {"dir": args.output_dir}
    config["evaluation"] = {
        "task": args.evaluation_task,
        "threshold": args.threshold,
        "tune_threshold_on_validation": args.tune_threshold_on_validation,
        "validation_split": args.validation_split,
    }

    evaluation = run_dataset(config, limit=args.limit)
    print(evaluation)


def run_dataset(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    data_root = Path(config["dataset"]["root"])
    dataset = config["dataset"]["name"]
    subset = config["dataset"]["subset"]
    split = config["dataset"].get("split")
    output_dir = Path(config["outputs"]["dir"])

    examples = load_examples(data_root, dataset, subset, split=split)
    metric = GraphMetric(config)
    evaluation_config = config.get("evaluation", {})
    task = evaluation_config.get("task", "correlation")
    threshold = float(evaluation_config.get("threshold", 0.5))
    validation_summary_rows: list[dict[str, Any]] = []
    threshold_selection = None

    if task == "binary" and evaluation_config.get("tune_threshold_on_validation", False):
        validation_split = evaluation_config.get("validation_split", "validation")
        validation_examples = load_examples(data_root, dataset, subset, split=validation_split)
        _, _, _, validation_summary_rows = metric.score_examples(validation_examples)
        threshold_selection = choose_best_binary_threshold(validation_summary_rows)
        threshold = float(threshold_selection["threshold"])

    claim_rows, graph_rows, sentence_rows, summary_rows = metric.score_examples(examples, limit=limit)
    evaluation = evaluate(summary_rows, task=task, threshold=threshold)
    evaluation.update(metadata(config, dataset, subset, split, examples, summary_rows))
    if threshold_selection is not None:
        evaluation["threshold_selection"] = threshold_selection
        evaluation["threshold_source_split"] = evaluation_config.get("validation_split", "validation")

    write_outputs(output_dir, claim_rows, graph_rows, sentence_rows, summary_rows, validation_summary_rows, evaluation)
    return evaluation


def metadata(
    config: dict[str, Any],
    dataset: str,
    subset: str,
    split: str | None,
    examples: list[Any],
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "n_total_examples": len(examples),
        "n_scored_examples": len(summary_rows),
        "claim_mode": config.get("claim_extraction", {}).get("mode", "sentence"),
        "claim_filter_with_nli": config.get("claim_extraction", {}).get("filter_with_nli", False),
        "claim_filter_threshold": config.get("claim_extraction", {}).get("filter_threshold"),
        "embedding_model": config["retrieval"]["embedding_model"],
        "retrieval_mode": config["retrieval"].get("mode", "dense"),
        "dense_weight": config["retrieval"].get("dense_weight"),
        "top_k": config["retrieval"]["top_k"],
        "neighbor_window": config["retrieval"]["neighbor_window"],
        "use_nli": config["scoring"]["use_nli"],
        "nli_model": config["scoring"].get("nli_model"),
        "use_bartscore": config["scoring"].get("use_bartscore", False),
        "bartscore_model": config["scoring"].get("bartscore_model"),
        "sentence_aggregation": config["scoring"].get("sentence_aggregation", "max"),
        "claim_aggregation": config["scoring"].get("claim_aggregation", "mean"),
        "summary_aggregation": config["scoring"].get("summary_aggregation", "sentence_mean"),
    }


def write_outputs(
    output_dir: Path,
    claim_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_summary_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> None:
    write_jsonl(output_dir / "graph_metric_claim_predictions.jsonl", claim_rows)
    write_jsonl(output_dir / "graph_metric_graphs.jsonl", graph_rows)
    write_jsonl(output_dir / "graph_metric_sentence_diagnostics.jsonl", sentence_rows)
    write_csv(output_dir / "graph_metric_sentence_diagnostics.csv", sentence_rows)
    write_csv(
        output_dir / "graph_metric_top_supported_sentences.csv",
        sorted(sentence_rows, key=lambda row: int(row["support_rank"])),
    )
    write_csv(
        output_dir / "graph_metric_failure_risk_sentences.csv",
        sorted(sentence_rows, key=lambda row: int(row["failure_risk_rank"])),
    )
    write_csv(
        output_dir / "graph_metric_summary_disagreement_sentences.csv",
        sorted(sentence_rows, key=lambda row: int(row["summary_disagreement_rank"])),
    )
    write_csv(output_dir / "graph_metric_summary_scores.csv", summary_rows)
    if validation_summary_rows:
        write_csv(output_dir / "graph_metric_validation_summary_scores.csv", validation_summary_rows)
    write_json(output_dir / "graph_metric_evaluation.json", evaluation)


if __name__ == "__main__":
    main()
