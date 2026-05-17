#!/usr/bin/env bash
set -uo pipefail

# Paper-oriented sweep for GraphLongDocFACTScore.
# It intentionally avoids a full Cartesian product. Each group isolates one
# scientific question while keeping the current best configuration fixed.

SWEEP_GROUP="${SWEEP_GROUP:-all}"
OUT_ROOT="${OUT_ROOT:-data/outputs/graph_metric_paper_sweep}"
RESULTS_CSV="${RESULTS_CSV:-$OUT_ROOT/sweep_results.csv}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
INCLUDE_HEAVY="${INCLUDE_HEAVY:-0}"
INCLUDE_ATOMIC="${INCLUDE_ATOMIC:-0}"
INCLUDE_SQUALITY="${INCLUDE_SQUALITY:-0}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TOKENIZERS_PARALLELISM

mkdir -p "$OUT_ROOT"
if [ ! -f "$RESULTS_CSV" ]; then
  printf '%s\n' 'experiment_id,dataset,subset,config,embedding_model,top_k,neighbor_window,nli_mode,nli_model,use_bartscore,weight_preset,sentence_aggregation,claim_mode,output_dir,kendall_tau,p_value,n,status' > "$RESULTS_CSV"
fi

DATASETS=(
  "longsciverify pubmed configs/graph_metric_pubmed.yaml"
  "longsciverify arxiv configs/graph_metric_arxiv.yaml"
  "longeval pubmed configs/graph_metric_longeval_pubmed.yaml"
)
if [ "$INCLUDE_SQUALITY" = "1" ]; then
  DATASETS+=("longeval squality configs/graph_metric_longeval_squality.yaml")
fi

sanitize() {
  printf '%s' "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

append_result() {
  local experiment_id="$1"
  local dataset="$2"
  local subset="$3"
  local config="$4"
  local embedding="$5"
  local top_k="$6"
  local window="$7"
  local nli_mode="$8"
  local nli_model="$9"
  local use_bartscore="${10}"
  local weight_preset="${11}"
  local sentence_aggregation="${12}"
  local claim_mode="${13}"
  local output_dir="${14}"
  local status="${15}"

  python - "$RESULTS_CSV" "$output_dir/graph_metric_evaluation.json" "$experiment_id" "$dataset" "$subset" "$config" "$embedding" "$top_k" "$window" "$nli_mode" "$nli_model" "$use_bartscore" "$weight_preset" "$sentence_aggregation" "$claim_mode" "$output_dir" "$status" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

(
    csv_path,
    eval_path,
    experiment_id,
    dataset,
    subset,
    config,
    embedding,
    top_k,
    window,
    nli_mode,
    nli_model,
    use_bartscore,
    weight_preset,
    sentence_aggregation,
    claim_mode,
    output_dir,
    status,
) = sys.argv[1:]

evaluation = {}
path = Path(eval_path)
if path.exists():
    with path.open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)

row = {
    "experiment_id": experiment_id,
    "dataset": dataset,
    "subset": subset,
    "config": config,
    "embedding_model": embedding,
    "top_k": top_k,
    "neighbor_window": window,
    "nli_mode": nli_mode,
    "nli_model": nli_model,
    "use_bartscore": use_bartscore,
    "weight_preset": weight_preset,
    "sentence_aggregation": sentence_aggregation,
    "claim_mode": claim_mode,
    "output_dir": output_dir,
    "kendall_tau": evaluation.get("kendall_tau", ""),
    "p_value": evaluation.get("p_value", ""),
    "n": evaluation.get("n", ""),
    "status": status,
}
with open(csv_path, "a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writerow(row)
PY
}

should_run_group() {
  local group="$1"
  [ "$SWEEP_GROUP" = "all" ] || [ "$SWEEP_GROUP" = "$group" ]
}

run_one() {
  local group="$1"
  local experiment_id="$2"
  local dataset="$3"
  local subset="$4"
  local config="$5"
  local embedding="$6"
  local top_k="$7"
  local window="$8"
  local nli_mode="$9"
  local nli_model="${10}"
  local use_bartscore="${11}"
  local weight_preset="${12}"
  local sentence_aggregation="${13}"
  local claim_mode="${14}"

  if ! should_run_group "$group"; then
    return 0
  fi

  local out_dir="$OUT_ROOT/$dataset/$subset/$(sanitize "$experiment_id")"
  if [ -f "$out_dir/graph_metric_evaluation.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP existing: $experiment_id [$dataset/$subset]"
    append_result "$experiment_id" "$dataset" "$subset" "$config" "$embedding" "$top_k" "$window" "$nli_mode" "$nli_model" "$use_bartscore" "$weight_preset" "$sentence_aggregation" "$claim_mode" "$out_dir" "cached"
    return 0
  fi

  echo
  echo "RUN $experiment_id [$dataset/$subset]"
  echo "  embedding=$embedding k=$top_k window=$window nli=$nli_mode bart=$use_bartscore preset=$weight_preset agg=$sentence_aggregation claim=$claim_mode"

  local cmd=(
    python graph_metric/run_graph_metric.py
    --config "$config"
    --dataset "$dataset"
    --subset "$subset"
    --embedding-model "$embedding"
    --top-k "$top_k"
    --neighbor-window "$window"
    --weight-preset "$weight_preset"
    --sentence-aggregation "$sentence_aggregation"
    --claim-mode "$claim_mode"
    --output-dir "$out_dir"
  )
  if [ "$use_bartscore" = "true" ]; then
    cmd+=(--use-bartscore)
  else
    cmd+=(--no-bartscore)
  fi
  if [ "$nli_mode" = "none" ]; then
    cmd+=(--no-nli)
  else
    cmd+=(--use-nli --nli-model "$nli_model")
  fi

  if "${cmd[@]}"; then
    append_result "$experiment_id" "$dataset" "$subset" "$config" "$embedding" "$top_k" "$window" "$nli_mode" "$nli_model" "$use_bartscore" "$weight_preset" "$sentence_aggregation" "$claim_mode" "$out_dir" "ok"
  else
    append_result "$experiment_id" "$dataset" "$subset" "$config" "$embedding" "$top_k" "$window" "$nli_mode" "$nli_model" "$use_bartscore" "$weight_preset" "$sentence_aggregation" "$claim_mode" "$out_dir" "failed"
    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
      return 1
    fi
  fi
}

run_for_all_datasets() {
  local group="$1"
  local experiment_id="$2"
  local embedding="$3"
  local top_k="$4"
  local window="$5"
  local nli_mode="$6"
  local nli_model="$7"
  local use_bartscore="$8"
  local weight_preset="$9"
  local sentence_aggregation="${10}"
  local claim_mode="${11}"

  for dataset_spec in "${DATASETS[@]}"; do
    read -r dataset subset config <<< "$dataset_spec"
    run_one "$group" "$experiment_id" "$dataset" "$subset" "$config" "$embedding" "$top_k" "$window" "$nli_mode" "$nli_model" "$use_bartscore" "$weight_preset" "$sentence_aggregation" "$claim_mode"
  done
}

BEST_EMBEDDING="sentence-transformers/bert-base-nli-mean-tokens"
BEST_NLI="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
BASE_NLI="cross-encoder/nli-deberta-v3-large"

# 1. Main model and paper-style control.
run_for_all_datasets core best_bert_k5_w0_wanli_graph_bart "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true graph_bart max sentence
run_for_all_datasets core paper_style_bert_k3_w1_wanli_graph_bart "$BEST_EMBEDDING" 3 1 wanli "$BEST_NLI" true graph_bart max sentence

# 2. Embedding-model ablation. Choices are motivated by SentenceTransformers docs:
# original LongDocFACTScore embedding, best-quality general model, fast general model,
# semantic-search QA models, and optional heavy/new or scientific-domain models.
EMBEDDINGS=(
  "sentence-transformers/bert-base-nli-mean-tokens"
  "sentence-transformers/all-mpnet-base-v2"
  "sentence-transformers/all-MiniLM-L6-v2"
  "sentence-transformers/multi-qa-mpnet-base-cos-v1"
  "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
  "sentence-transformers/allenai-specter"
)
if [ "$INCLUDE_HEAVY" = "1" ]; then
  EMBEDDINGS+=("Qwen/Qwen3-Embedding-0.6B")
fi
for embedding in "${EMBEDDINGS[@]}"; do
  run_for_all_datasets embedding "embed_$(sanitize "$embedding")" "$embedding" 5 0 wanli "$BEST_NLI" true graph_bart max sentence
done

# 3. Retrieval-depth and context-window ablations.
for top_k in 3 5 8; do
  run_for_all_datasets retrieval "retrieval_k${top_k}_w0" "$BEST_EMBEDDING" "$top_k" 0 wanli "$BEST_NLI" true graph_bart max sentence
done
for window in 0 1 2; do
  run_for_all_datasets retrieval "retrieval_k5_w${window}" "$BEST_EMBEDDING" 5 "$window" wanli "$BEST_NLI" true graph_bart max sentence
done

# 4. NLI and score-component ablations.
run_for_all_datasets nli nli_wanli "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true graph_bart max sentence
run_for_all_datasets nli nli_cross_encoder_deberta_large "$BEST_EMBEDDING" 5 0 cross_encoder "$BASE_NLI" true graph_bart max sentence
run_for_all_datasets nli no_nli_graph_bart "$BEST_EMBEDDING" 5 0 none none true graph_bart max sentence
run_for_all_datasets components no_bartscore_retrieval_heavy "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" false retrieval_heavy max sentence
run_for_all_datasets components bartscore_heavy "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true bartscore_heavy max sentence
run_for_all_datasets components balanced_graph "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true balanced max sentence

# 5. Aggregation ablations.
for aggregation in max softmax top2_mean top3_mean rank_weighted; do
  run_for_all_datasets aggregation "sentagg_${aggregation}" "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true graph_bart "$aggregation" sentence
done

# 6. Optional claim granularity ablation. This is slower because it loads the local claim extractor.
if [ "$INCLUDE_ATOMIC" = "1" ]; then
  run_for_all_datasets claim atomic_best "$BEST_EMBEDDING" 5 0 wanli "$BEST_NLI" true graph_bart max atomic
fi

echo
echo "Sweep results: $RESULTS_CSV"
