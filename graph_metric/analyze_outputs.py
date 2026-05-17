from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze graph metric outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    claims_path = output_dir / "graph_metric_claim_predictions.jsonl"
    summary_path = output_dir / "graph_metric_summary_scores.csv"
    eval_path = output_dir / "graph_metric_evaluation.json"

    claim_rows = read_jsonl(claims_path)
    summary_rows = read_csv(summary_path)
    evaluation = read_json(eval_path)

    by_sentence: dict[str, list[dict]] = defaultdict(list)
    unchanged = 0
    entailments: list[float] = []
    for row in claim_rows:
        by_sentence[row["sentence_id"]].append(row)
        if normalize(row.get("claim_text", "")) == normalize(row.get("parent_sentence_text", "")):
            unchanged += 1
        value = row.get("summary_claim_entailment")
        if value is not None:
            entailments.append(float(value))

    claim_counts = Counter(len(rows) for rows in by_sentence.values())
    total_claims = len(claim_rows)
    total_sentences = len(by_sentence)
    changed = total_claims - unchanged

    print("evaluation:", evaluation)
    print("summaries:", len(summary_rows))
    print("sentences:", total_sentences)
    print("claims:", total_claims)
    print("claims_per_sentence:", dict(sorted(claim_counts.items())))
    print("unchanged_claims:", unchanged)
    print("changed_claims:", changed)
    print("changed_claim_rate:", round(changed / total_claims, 4) if total_claims else 0.0)
    if entailments:
        print("summary_claim_entailment_mean:", round(mean(entailments), 4))
        print("summary_claim_entailment_min:", round(min(entailments), 4))
        print("summary_claim_entailment_max:", round(max(entailments), 4))

    print("\nexamples:")
    shown = 0
    for sentence_id, rows in by_sentence.items():
        if shown >= args.examples:
            break
        parent = rows[0].get("parent_sentence_text", "")
        claims = [row.get("claim_text", "") for row in rows]
        if len(claims) > 1 or any(normalize(claim) != normalize(parent) for claim in claims):
            print(f"\n{sentence_id}")
            print("SENT:", parent)
            for row in rows:
                print(
                    "CLAIM:",
                    row.get("claim_text", ""),
                    "score=",
                    round(float(row.get("support_score", 0.0)), 4),
                    "summary_entailment=",
                    row.get("summary_claim_entailment"),
                )
            shown += 1

    if shown == 0:
        print("No changed/decomposed claims found in the first pass.")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


if __name__ == "__main__":
    main()

