from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_metric.data_io import load_config, load_examples
from graph_metric.metric import GraphMetric
from graph_metric.report import choose_best_binary_threshold, evaluate, write_csv, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local graph metric on long-document factuality datasets.")
    parser.add_argument("--config", default="configs/graph_metric_pubmed.yaml")
    parser.add_argument("--dataset", choices=["longsciverify", "longeval", "fenice", "screen_eval"], default=None)
    parser.add_argument("--subset", choices=["pubmed", "arxiv", "squality", "story_summeval", "dialogue"], default=None)
    parser.add_argument("--split", choices=["validation", "test", "all"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--claim-mode", choices=["sentence", "atomic"], default=None)
    parser.add_argument("--claim-extractor-model", default=None)
    parser.add_argument("--claim-filter-threshold", type=float, default=None)
    parser.add_argument("--no-claim-filter", action="store_true")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--retrieval-mode", choices=["dense", "lexical", "hybrid"], default=None)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--neighbor-window", type=int, default=None)
    parser.add_argument(
        "--sentence-aggregation",
        choices=["max", "top2_mean", "top3_mean", "rank_weighted", "softmax"],
        default=None,
    )
    parser.add_argument(
        "--claim-aggregation",
        choices=["mean", "min", "bottom2_mean"],
        default=None,
    )
    parser.add_argument(
        "--summary-aggregation",
        choices=["sentence_mean", "claim_mean"],
        default=None,
    )
    parser.add_argument("--use-nli", action="store_true", default=None)
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--use-bartscore", action="store_true", default=None)
    parser.add_argument("--no-bartscore", action="store_true")
    parser.add_argument("--bartscore-model", default=None)
    parser.add_argument("--bartscore-batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tune-threshold-on-validation", action="store_true")
    parser.add_argument(
        "--weight-preset",
        choices=["balanced", "nli_heavy", "retrieval_heavy", "numeric_strict", "bartscore_heavy", "graph_bart"],
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_overrides(config, args)
    data_root = Path(config["dataset"]["root"])
    dataset = config["dataset"].get("name", "longsciverify")
    subset = config["dataset"].get("subset", "pubmed")
    split = config["dataset"].get("split")
    output_dir = Path(config["outputs"]["dir"])

    examples = load_examples(data_root, dataset, subset, split=split)
    metric = GraphMetric(config)
    validation_summary_rows = []
    threshold_selection = None
    evaluation_config = config.get("evaluation", {})
    task = evaluation_config.get("task", "correlation")
    threshold = float(evaluation_config.get("threshold", 0.5))
    if task == "binary" and evaluation_config.get("tune_threshold_on_validation", False):
        validation_split = evaluation_config.get("validation_split", "validation")
        validation_examples = load_examples(data_root, dataset, subset, split=validation_split)
        _, _, _, validation_summary_rows = metric.score_examples(validation_examples)
        threshold_selection = choose_best_binary_threshold(validation_summary_rows)
        threshold = float(threshold_selection["threshold"])

    claim_rows, graph_rows, sentence_rows, summary_rows = metric.score_examples(examples, limit=args.limit)
    evaluation = evaluate(summary_rows, task=task, threshold=threshold)
    evaluation.update(
        {
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
    )
    if threshold_selection is not None:
        evaluation["threshold_selection"] = threshold_selection
        evaluation["threshold_source_split"] = evaluation_config.get("validation_split", "validation")

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
    print(evaluation)


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    config.setdefault("dataset", {})
    config.setdefault("claim_extraction", {})
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.subset:
        config["dataset"]["subset"] = args.subset
    if args.split:
        config["dataset"]["split"] = args.split
    if args.claim_mode:
        config["claim_extraction"]["mode"] = args.claim_mode
    if args.claim_extractor_model:
        config["claim_extraction"]["model"] = args.claim_extractor_model
    if args.no_claim_filter:
        config["claim_extraction"]["filter_with_nli"] = False
    if args.claim_filter_threshold is not None:
        config["claim_extraction"]["filter_threshold"] = args.claim_filter_threshold
    if args.embedding_model:
        config["retrieval"]["embedding_model"] = args.embedding_model
    if args.retrieval_mode:
        config["retrieval"]["mode"] = args.retrieval_mode
    if args.dense_weight is not None:
        config["retrieval"]["dense_weight"] = args.dense_weight
    if args.top_k is not None:
        config["retrieval"]["top_k"] = args.top_k
    if args.neighbor_window is not None:
        config["retrieval"]["neighbor_window"] = args.neighbor_window
    if args.sentence_aggregation:
        config["scoring"]["sentence_aggregation"] = args.sentence_aggregation
    if args.claim_aggregation:
        config["scoring"]["claim_aggregation"] = args.claim_aggregation
    if args.summary_aggregation:
        config["scoring"]["summary_aggregation"] = args.summary_aggregation
    if args.no_nli:
        config["scoring"]["use_nli"] = False
    elif args.use_nli:
        config["scoring"]["use_nli"] = True
    if args.nli_model:
        config["scoring"]["nli_model"] = args.nli_model
    if args.no_bartscore:
        config["scoring"]["use_bartscore"] = False
    elif args.use_bartscore:
        config["scoring"]["use_bartscore"] = True
    if args.bartscore_model:
        config["scoring"]["bartscore_model"] = args.bartscore_model
    if args.bartscore_batch_size is not None:
        config["scoring"]["bartscore_batch_size"] = args.bartscore_batch_size
    if args.threshold is not None:
        config.setdefault("evaluation", {})
        config["evaluation"]["threshold"] = args.threshold
    if args.tune_threshold_on_validation:
        config.setdefault("evaluation", {})
        config["evaluation"]["tune_threshold_on_validation"] = True
    if args.weight_preset:
        config["scoring"]["weights"] = weight_preset(args.weight_preset)
    if args.output_dir:
        config["outputs"]["dir"] = args.output_dir


def weight_preset(name: str) -> dict[str, float]:
    presets = {
        "balanced": {
            "nli_support": 0.35,
            "bartscore_support": 0.0,
            "retrieval_score": 0.20,
            "lexical_anchor_coverage": 0.15,
            "evidence_precision": 0.05,
            "numeric_match": 0.15,
            "polarity_consistency": 0.10,
            "section_relevance": 0.05,
            "numeric_mismatch": -0.35,
            "missing_anchor_penalty": -0.15,
        },
        "nli_heavy": {
            "nli_support": 0.55,
            "bartscore_support": 0.0,
            "retrieval_score": 0.10,
            "lexical_anchor_coverage": 0.10,
            "evidence_precision": 0.03,
            "numeric_match": 0.12,
            "polarity_consistency": 0.08,
            "section_relevance": 0.02,
            "numeric_mismatch": -0.35,
            "missing_anchor_penalty": -0.10,
        },
        "retrieval_heavy": {
            "nli_support": 0.20,
            "bartscore_support": 0.0,
            "retrieval_score": 0.35,
            "lexical_anchor_coverage": 0.20,
            "evidence_precision": 0.05,
            "numeric_match": 0.10,
            "polarity_consistency": 0.07,
            "section_relevance": 0.03,
            "numeric_mismatch": -0.25,
            "missing_anchor_penalty": -0.18,
        },
        "numeric_strict": {
            "nli_support": 0.30,
            "bartscore_support": 0.0,
            "retrieval_score": 0.17,
            "lexical_anchor_coverage": 0.14,
            "evidence_precision": 0.04,
            "numeric_match": 0.25,
            "polarity_consistency": 0.10,
            "section_relevance": 0.03,
            "numeric_mismatch": -0.60,
            "missing_anchor_penalty": -0.15,
        },
        "bartscore_heavy": {
            "nli_support": 0.08,
            "bartscore_support": 0.42,
            "retrieval_score": 0.22,
            "lexical_anchor_coverage": 0.12,
            "evidence_precision": 0.03,
            "numeric_match": 0.08,
            "polarity_consistency": 0.04,
            "section_relevance": 0.01,
            "numeric_mismatch": -0.25,
            "missing_anchor_penalty": -0.12,
        },
        "graph_bart": {
            "nli_support": 0.12,
            "bartscore_support": 0.25,
            "retrieval_score": 0.28,
            "lexical_anchor_coverage": 0.16,
            "evidence_precision": 0.04,
            "numeric_match": 0.09,
            "polarity_consistency": 0.04,
            "section_relevance": 0.02,
            "numeric_mismatch": -0.25,
            "missing_anchor_penalty": -0.15,
        },
    }
    return presets[name]


if __name__ == "__main__":
    main()
