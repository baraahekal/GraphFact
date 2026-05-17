from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two graph metric output directories.")
    parser.add_argument("--a", required=True, help="First output directory, usually current best sentence run.")
    parser.add_argument("--b", required=True, help="Second output directory, usually atomic run.")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    summaries_a = keyed(read_csv(Path(args.a) / "graph_metric_summary_scores.csv"))
    summaries_b = keyed(read_csv(Path(args.b) / "graph_metric_summary_scores.csv"))
    claims_a = claims_by_summary(read_jsonl(Path(args.a) / "graph_metric_claim_predictions.jsonl"))
    claims_b = claims_by_summary(read_jsonl(Path(args.b) / "graph_metric_claim_predictions.jsonl"))

    rows = []
    for key in sorted(set(summaries_a) & set(summaries_b)):
        row_a = summaries_a[key]
        row_b = summaries_b[key]
        human = float(row_a["human_summary_score"])
        score_a = float(row_a["graph_summary_score"])
        score_b = float(row_b["graph_summary_score"])
        rows.append(
            {
                "key": key,
                "human": human,
                "a": score_a,
                "b": score_b,
                "a_abs_error": abs(score_a - human),
                "b_abs_error": abs(score_b - human),
                "delta_b_minus_a": score_b - score_a,
                "b_error_minus_a_error": abs(score_b - human) - abs(score_a - human),
                "a_claim_stats": claim_stats(claims_a.get(key, [])),
                "b_claim_stats": claim_stats(claims_b.get(key, [])),
            }
        )

    print("count:", len(rows))
    print(args.a_name, "mean_abs_error:", round(mean(row["a_abs_error"] for row in rows), 4))
    print(args.b_name, "mean_abs_error:", round(mean(row["b_abs_error"] for row in rows), 4))
    print(args.b_name, "better_count:", sum(row["b_abs_error"] < row["a_abs_error"] for row in rows))
    print(args.a_name, "better_count:", sum(row["a_abs_error"] < row["b_abs_error"] for row in rows))
    print("ties:", sum(row["a_abs_error"] == row["b_abs_error"] for row in rows))

    print_section(
        f"{args.b_name} helps most",
        sorted(rows, key=lambda row: row["b_error_minus_a_error"])[: args.top],
        args,
    )
    print_section(
        f"{args.b_name} hurts most",
        sorted(rows, key=lambda row: row["b_error_minus_a_error"], reverse=True)[: args.top],
        args,
    )
    print_section(
        f"{args.a_name} high on bad human summaries",
        sorted(rows, key=lambda row: (row["human"], -row["a"]))[: args.top],
        args,
    )
    print_section(
        f"{args.a_name} low on good human summaries",
        sorted(rows, key=lambda row: (-row["human"], row["a"]))[: args.top],
        args,
    )


def print_section(title: str, rows: list[dict], args: argparse.Namespace) -> None:
    print(f"\n## {title}")
    for row in rows:
        print(
            row["key"],
            "human=",
            round(row["human"], 4),
            args.a_name + "=",
            round(row["a"], 4),
            args.b_name + "=",
            round(row["b"], 4),
            "delta=",
            round(row["delta_b_minus_a"], 4),
            "err_delta=",
            round(row["b_error_minus_a_error"], 4),
        )
        print(" ", args.a_name, row["a_claim_stats"])
        print(" ", args.b_name, row["b_claim_stats"])


def claim_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    scores = [float(row.get("support_score", 0.0)) for row in rows]
    return {
        "claims": len(rows),
        "mean": round(mean(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "low_lt_0.4": sum(score < 0.4 for score in scores),
        "high_gt_0.8": sum(score > 0.8 for score in scores),
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


if __name__ == "__main__":
    main()

