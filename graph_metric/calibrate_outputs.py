from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "base_graph_summary_score",
    "num_summary_sentences",
    "num_claims",
    "claim_score_mean",
    "claim_score_min",
    "claim_score_max",
    "claim_score_std",
    "claim_low_fraction",
    "claim_high_fraction",
    "best_graph_score_mean",
    "best_graph_score_min",
    "feature_retrieval_score_mean",
    "feature_dense_retrieval_score_mean",
    "feature_lexical_retrieval_score_mean",
    "feature_nli_support_mean",
    "feature_bartscore_support_mean",
    "feature_lexical_anchor_coverage_mean",
    "feature_evidence_precision_mean",
    "feature_numeric_match_mean",
    "feature_numeric_mismatch_mean",
    "feature_polarity_consistency_mean",
    "feature_missing_anchor_penalty_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate lightweight calibrators over graph metric outputs.")
    parser.add_argument("--output-dir", required=True, help="Graph metric output directory to calibrate.")
    parser.add_argument("--cv", choices=["leave_document_out", "leave_summary_out"], default="leave_document_out")
    parser.add_argument("--target", choices=["human_summary_score"], default="human_summary_score")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    examples = build_examples(output_dir)
    if len(examples) < 5:
        raise RuntimeError(f"Need at least 5 examples, found {len(examples)}")

    results = []
    for model_name, factory in model_factories().items():
        predictions = cross_validated_predictions(examples, factory, args.cv)
        human = np.array([example["target"] for example in examples], dtype=float)
        tau, p_value = kendalltau(human, predictions)
        mae = float(np.mean(np.abs(predictions - human)))
        results.append(
            {
                "model": model_name,
                "cv": args.cv,
                "kendall_tau": float(tau),
                "p_value": float(p_value),
                "mean_abs_error": mae,
                "n": len(examples),
            }
        )

    base_scores = np.array([example["features"][0] for example in examples], dtype=float)
    human = np.array([example["target"] for example in examples], dtype=float)
    base_tau, base_p = kendalltau(human, base_scores)
    results.append(
        {
            "model": "base_graph_score",
            "cv": "none",
            "kendall_tau": float(base_tau),
            "p_value": float(base_p),
            "mean_abs_error": float(np.mean(np.abs(base_scores - human))),
            "n": len(examples),
        }
    )

    results.sort(key=lambda row: row["kendall_tau"], reverse=True)
    payload = {
        "output_dir": str(output_dir),
        "feature_names": FEATURE_NAMES,
        "results": results,
        "examples": [
            {
                "document_id": example["document_id"],
                "summary_model": example["summary_model"],
                "human": example["target"],
                "base_graph_score": example["features"][0],
            }
            for example in examples
        ],
    }

    for row in results:
        print(row)

    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def build_examples(output_dir: Path) -> list[dict[str, Any]]:
    summary_rows = keyed(read_csv(output_dir / "graph_metric_summary_scores.csv"))
    claim_rows = claims_by_summary(read_jsonl(output_dir / "graph_metric_claim_predictions.jsonl"))

    examples = []
    for key, row in sorted(summary_rows.items()):
        document_id, summary_model = key
        claims = claim_rows.get(key, [])
        features = summary_features(row, claims)
        examples.append(
            {
                "document_id": document_id,
                "summary_model": summary_model,
                "target": float(row["human_summary_score"]),
                "features": np.array(features, dtype=float),
            }
        )
    return examples


def summary_features(summary_row: dict[str, str], claims: list[dict[str, Any]]) -> list[float]:
    claim_scores = [float(row.get("support_score", 0.0)) for row in claims]
    best_graph_scores = [float(row.get("best_graph_score", row.get("support_score", 0.0))) for row in claims]

    values = [
        float(summary_row["graph_summary_score"]),
        float(summary_row.get("num_summary_sentences", 0) or 0),
        float(summary_row.get("num_claims", len(claims)) or len(claims)),
        safe_mean(claim_scores),
        safe_min(claim_scores),
        safe_max(claim_scores),
        safe_std(claim_scores),
        fraction(claim_scores, lambda value: value < 0.4),
        fraction(claim_scores, lambda value: value > 0.8),
        safe_mean(best_graph_scores),
        safe_min(best_graph_scores),
    ]

    feature_keys = [
        "retrieval_score",
        "dense_retrieval_score",
        "lexical_retrieval_score",
        "nli_support",
        "bartscore_support",
        "lexical_anchor_coverage",
        "evidence_precision",
        "numeric_match",
        "numeric_mismatch",
        "polarity_consistency",
        "missing_anchor_penalty",
    ]
    for key in feature_keys:
        values.append(safe_mean([float(row.get("features", {}).get(key, 0.0)) for row in claims]))
    return values


def cross_validated_predictions(examples: list[dict[str, Any]], factory, cv: str) -> np.ndarray:
    predictions = np.zeros(len(examples), dtype=float)
    groups = defaultdict(list)
    for idx, example in enumerate(examples):
        group_key = example["document_id"] if cv == "leave_document_out" else (example["document_id"], example["summary_model"])
        groups[group_key].append(idx)

    x = np.vstack([example["features"] for example in examples])
    y = np.array([example["target"] for example in examples], dtype=float)
    for test_indices in groups.values():
        train_indices = [idx for idx in range(len(examples)) if idx not in test_indices]
        model = factory()
        model.fit(x[train_indices], y[train_indices])
        predictions[test_indices] = np.clip(model.predict(x[test_indices]), 0.0, 1.0)
    return predictions


def model_factories():
    return {
        "ridge_alpha_0.1": lambda: make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "ridge_alpha_1": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_alpha_10": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "elasticnet_alpha_0.01": lambda: make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20000, random_state=0)
        ),
        "elasticnet_alpha_0.05": lambda: make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20000, random_state=0)
        ),
        "random_forest_small": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=3,
            random_state=0,
        ),
    }


def keyed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["document_id"], row["summary_model"]): row for row in rows}


def claims_by_summary(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["document_id"], row["summary_model"])].append(row)
    return grouped


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_min(values: list[float]) -> float:
    return float(min(values)) if values else 0.0


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def fraction(values: list[float], predicate) -> float:
    return float(sum(1 for value in values if predicate(value)) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()

