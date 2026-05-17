from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scipy.stats import kendalltau


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def evaluate(
    summary_rows: list[dict[str, Any]],
    task: str = "correlation",
    threshold: float | None = None,
) -> dict[str, Any]:
    if task == "binary":
        return evaluate_binary(summary_rows, threshold=0.5 if threshold is None else threshold)
    if task != "correlation":
        raise ValueError(f"Unsupported evaluation task: {task}")
    human = [float(row["human_summary_score"]) for row in summary_rows]
    graph = [float(row["graph_summary_score"]) for row in summary_rows]
    tau, p_value = kendalltau(human, graph)
    return {
        "metric": "GraphLongDocFACTScore",
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "n": len(summary_rows),
    }


def evaluate_binary(summary_rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    labels = [1 if float(row["human_summary_score"]) >= 0.5 else 0 for row in summary_rows]
    scores = [float(row["graph_summary_score"]) for row in summary_rows]
    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for gold, pred in zip(labels, predictions) if gold == 1 and pred == 1)
    tn = sum(1 for gold, pred in zip(labels, predictions) if gold == 0 and pred == 0)
    fp = sum(1 for gold, pred in zip(labels, predictions) if gold == 0 and pred == 1)
    fn = sum(1 for gold, pred in zip(labels, predictions) if gold == 1 and pred == 0)

    positive = tp + fn
    negative = tn + fp
    recall = tp / positive if positive else float("nan")
    specificity = tn / negative if negative else float("nan")
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = (tp + tn) / len(labels) if labels else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    roc_auc = compute_roc_auc(labels, scores)

    return {
        "metric": "GraphLongDocFACTScore",
        "evaluation_task": "binary_factuality",
        "primary_metric": "balanced_accuracy",
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "n": len(summary_rows),
        "n_positive": int(positive),
        "n_negative": int(negative),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def choose_best_binary_threshold(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted({float(row["graph_summary_score"]) for row in summary_rows})
    if not scores:
        return {"threshold": 0.5, "balanced_accuracy": float("nan")}
    candidates = [max(0.0, scores[0] - 1e-9), min(1.0, scores[-1] + 1e-9)]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    best = None
    for threshold in candidates:
        evaluation = evaluate_binary(summary_rows, threshold=threshold)
        current = {
            "threshold": float(threshold),
            "balanced_accuracy": evaluation["balanced_accuracy"],
            "accuracy": evaluation["accuracy"],
            "f1": evaluation["f1"],
        }
        if best is None or current["balanced_accuracy"] > best["balanced_accuracy"]:
            best = current
    return best or {"threshold": 0.5, "balanced_accuracy": float("nan")}


def compute_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    return float(roc_auc_score(labels, scores))
