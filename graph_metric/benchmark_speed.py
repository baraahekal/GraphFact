from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_metric.data_io import load_config, load_examples
from graph_metric.metric import GraphMetric
from graph_metric.report import write_json
from graph_metric.run_graph_metric import weight_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GraphLongDocFACTScore runtime.")
    parser.add_argument("--config", default="configs/graph_metric_pubmed.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--neighbor-window", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--use-bartscore", action="store_true")
    parser.add_argument("--no-bartscore", action="store_true")
    parser.add_argument("--weight-preset", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    apply_overrides(base_config, args)

    data_start = time.perf_counter()
    data_root = Path(base_config["dataset"]["root"])
    dataset = base_config["dataset"].get("name", "longsciverify")
    subset = base_config["dataset"].get("subset", "pubmed")
    split = base_config["dataset"].get("split")
    examples = load_examples(data_root, dataset, subset, split=split)
    selected = select_examples(examples, args.limit, args.sample_mode, args.seed)
    data_seconds = time.perf_counter() - data_start

    runs: list[dict[str, Any]] = []
    for run_idx in range(args.repeat):
        config = deepcopy(base_config)
        init_start = time.perf_counter()
        metric = GraphMetric(config)
        model_init_seconds = time.perf_counter() - init_start

        score_start = time.perf_counter()
        claim_rows, graph_rows, sentence_rows, summary_rows = metric.score_examples(selected)
        scoring_seconds = time.perf_counter() - score_start
        total_seconds = model_init_seconds + scoring_seconds

        runs.append(
            {
                "run_index": run_idx + 1,
                "model_init_seconds": model_init_seconds,
                "scoring_seconds": scoring_seconds,
                "total_model_plus_scoring_seconds": total_seconds,
                "n_examples": len(summary_rows),
                "n_summary_sentences": len(sentence_rows),
                "n_claims": len(claim_rows),
                "n_graph_candidates": len(graph_rows),
                "scoring_seconds_per_example": scoring_seconds / len(summary_rows) if summary_rows else None,
                "scoring_seconds_per_sentence": scoring_seconds / len(sentence_rows) if sentence_rows else None,
                "scoring_seconds_per_claim": scoring_seconds / len(claim_rows) if claim_rows else None,
            }
        )

    result = {
        "metric": "GraphLongDocFACTScore",
        "benchmark_style": f"LongDocFACTScore paper-style runtime on {args.sample_mode} N samples",
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "limit": args.limit,
        "sample_mode": args.sample_mode,
        "seed": args.seed if args.sample_mode == "random" else None,
        "sampled_examples": [
            {
                "document_id": example.document_id,
                "summary_model": example.summary_model,
            }
            for example in selected
        ],
        "repeat": args.repeat,
        "data_load_seconds": data_seconds,
        "config": {
            "embedding_model": base_config["retrieval"]["embedding_model"],
            "retrieval_mode": base_config["retrieval"].get("mode", "dense"),
            "top_k": base_config["retrieval"]["top_k"],
            "neighbor_window": base_config["retrieval"]["neighbor_window"],
            "use_nli": base_config["scoring"].get("use_nli", True),
            "nli_model": base_config["scoring"].get("nli_model"),
            "use_bartscore": base_config["scoring"].get("use_bartscore", False),
            "bartscore_model": base_config["scoring"].get("bartscore_model"),
            "sentence_aggregation": base_config["scoring"].get("sentence_aggregation", "max"),
            "claim_mode": base_config.get("claim_extraction", {}).get("mode", "sentence"),
            "claim_aggregation": base_config["scoring"].get("claim_aggregation", "mean"),
            "summary_aggregation": base_config["scoring"].get("summary_aggregation", "sentence_mean"),
        },
        "runs": runs,
        "summary": summarize_runs(runs),
    }

    if args.output_path:
        write_json(args.output_path, result)
    print(json.dumps(result, indent=2))


def select_examples(examples: list[Any], limit: int | None, sample_mode: str, seed: int) -> list[Any]:
    if not limit:
        return list(examples)
    if limit > len(examples):
        raise ValueError(f"Cannot select {limit} examples from only {len(examples)} loaded examples")
    if sample_mode == "first":
        return list(examples[:limit])
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config.setdefault("dataset", {})
    config.setdefault("retrieval", {})
    config.setdefault("scoring", {})
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.subset:
        config["dataset"]["subset"] = args.subset
    if args.split:
        config["dataset"]["split"] = args.split
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    if args.embedding_model:
        config["retrieval"]["embedding_model"] = args.embedding_model
    if args.all_candidates:
        config["retrieval"]["top_k"] = 10**9
    elif args.top_k is not None:
        config["retrieval"]["top_k"] = args.top_k
    if args.neighbor_window is not None:
        config["retrieval"]["neighbor_window"] = args.neighbor_window
    if args.nli_model:
        config["scoring"]["nli_model"] = args.nli_model
    if args.no_nli:
        config["scoring"]["use_nli"] = False
    if args.use_bartscore:
        config["scoring"]["use_bartscore"] = True
    if args.no_bartscore:
        config["scoring"]["use_bartscore"] = False
    if args.weight_preset:
        config["scoring"]["weights"] = weight_preset(args.weight_preset)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "model_init_seconds",
        "scoring_seconds",
        "total_model_plus_scoring_seconds",
        "scoring_seconds_per_example",
        "scoring_seconds_per_sentence",
        "scoring_seconds_per_claim",
    ]
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        if not values:
            continue
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    if runs:
        summary["n_examples"] = runs[-1]["n_examples"]
        summary["n_summary_sentences"] = runs[-1]["n_summary_sentences"]
        summary["n_claims"] = runs[-1]["n_claims"]
        summary["n_graph_candidates"] = runs[-1]["n_graph_candidates"]
    return summary


if __name__ == "__main__":
    main()
