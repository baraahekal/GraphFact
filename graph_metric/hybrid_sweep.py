from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
from scipy.stats import kendalltau


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hybrid score rules over sentence and atomic graph outputs.")
    parser.add_argument("--sentence-dir", required=True)
    parser.add_argument("--atomic-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    sentence_rows = keyed(read_csv(Path(args.sentence_dir) / "graph_metric_summary_scores.csv"))
    atomic_rows = keyed(read_csv(Path(args.atomic_dir) / "graph_metric_summary_scores.csv"))
    sentence_claims = claims_by_summary(read_jsonl(Path(args.sentence_dir) / "graph_metric_claim_predictions.jsonl"))
    atomic_claims = claims_by_summary(read_jsonl(Path(args.atomic_dir) / "graph_metric_claim_predictions.jsonl"))

    examples = []
    for key in sorted(set(sentence_rows) & set(atomic_rows)):
        human = float(sentence_rows[key]["human_summary_score"])
        sentence_score = float(sentence_rows[key]["graph_summary_score"])
        atomic_score = float(atomic_rows[key]["graph_summary_score"])
        examples.append(
            {
                "key": key,
                "human": human,
                "sentence": sentence_score,
                "atomic": atomic_score,
                "sentence_stats": claim_stats(sentence_claims.get(key, [])),
                "atomic_stats": claim_stats(atomic_claims.get(key, [])),
            }
        )

    results = []
    for name, fn in hybrid_rules():
        scores = [fn(example) for example in examples]
        human = [example["human"] for example in examples]
        tau, p_value = kendalltau(human, scores)
        mae = mean(abs(score - target) for score, target in zip(scores, human))
        results.append(
            {
                "rule": name,
                "kendall_tau": float(tau),
                "p_value": float(p_value),
                "mean_abs_error": float(mae),
                "n": len(examples),
            }
        )

    results.sort(key=lambda row: row["kendall_tau"], reverse=True)
    for row in results[: args.top]:
        print(row)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


def hybrid_rules() -> list[tuple[str, Callable[[dict], float]]]:
    rules: list[tuple[str, Callable[[dict], float]]] = [
        ("sentence", lambda ex: ex["sentence"]),
        ("atomic", lambda ex: ex["atomic"]),
        ("min_sentence_atomic", lambda ex: min(ex["sentence"], ex["atomic"])),
        ("max_sentence_atomic", lambda ex: max(ex["sentence"], ex["atomic"])),
        ("mean_sentence_atomic", lambda ex: 0.5 * ex["sentence"] + 0.5 * ex["atomic"]),
    ]

    for alpha in np.linspace(0.1, 0.9, 9):
        rules.append(
            (
                f"blend_sentence_{alpha:.1f}",
                lambda ex, alpha=alpha: alpha * ex["sentence"] + (1.0 - alpha) * ex["atomic"],
            )
        )

    for threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        rules.append(
            (
                f"use_atomic_when_lower_by_{threshold:.2f}",
                lambda ex, threshold=threshold: ex["atomic"]
                if ex["sentence"] - ex["atomic"] >= threshold
                else ex["sentence"],
            )
        )

    for low_fraction in [0.05, 0.10, 0.15, 0.20, 0.25]:
        for threshold in [0.02, 0.05, 0.08]:
            rules.append(
                (
                    f"atomic_lower_and_lowfrac_{low_fraction:.2f}_delta_{threshold:.2f}",
                    lambda ex, low_fraction=low_fraction, threshold=threshold: ex["atomic"]
                    if (
                        ex["sentence"] - ex["atomic"] >= threshold
                        and ex["atomic_stats"]["low_fraction"] >= low_fraction
                    )
                    else ex["sentence"],
                )
            )

    for penalty in [0.05, 0.10, 0.15, 0.20, 0.25]:
        rules.append(
            (
                f"sentence_minus_atomic_lowfrac_penalty_{penalty:.2f}",
                lambda ex, penalty=penalty: clamp01(ex["sentence"] - penalty * ex["atomic_stats"]["low_fraction"]),
            )
        )

    for penalty in [0.05, 0.10, 0.15, 0.20]:
        rules.append(
            (
                f"sentence_minus_atomic_lowfrac_if_many_claims_{penalty:.2f}",
                lambda ex, penalty=penalty: clamp01(
                    ex["sentence"] - penalty * ex["atomic_stats"]["low_fraction"]
                    if ex["atomic_stats"]["claims"] >= ex["sentence_stats"]["claims"] + 4
                    else ex["sentence"]
                ),
            )
        )

    return rules


def claim_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"claims": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "low_fraction": 0.0, "high_fraction": 0.0}
    scores = [float(row.get("support_score", 0.0)) for row in rows]
    return {
        "claims": len(rows),
        "mean": float(mean(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "low_fraction": sum(score < 0.4 for score in scores) / len(scores),
        "high_fraction": sum(score > 0.8 for score in scores) / len(scores),
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    main()

