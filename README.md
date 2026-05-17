# GraphLongDocFACTScore

GraphLongDocFACTScore is a local graph-guided factuality metric for source-grounded summarization. It scores a generated summary against the full source document and returns both a summary-level factuality score and an interpretable claim-evidence graph.

The metric does not use API calls, learned graph neural networks, or training loops. It builds an explicit graph between summary sentence claims and retrieved source evidence sentences, then scores each graph with retrieval, entailment, lexical, numeric, polarity, and section-relevance features.

## Install

```bash
git clone --recurse-submodules https://github.com/baraahekal/GraphFact.git
cd GraphFact
python -m pip install -r requirements.txt
```

If you cloned without submodules, fetch the benchmark data with:

```bash
git submodule update --init --recursive
```

For GPU use, install the PyTorch build that matches your CUDA environment before installing the remaining requirements.

## Quick Start

Score one summary against one full source document:

```bash
python -m graph_metric.score \
  --source article.txt \
  --summary summary.txt \
  --output result.json
```

Python API:

```python
from graph_metric import GraphMetric

metric = GraphMetric()
result = metric.score(
    source_doc=open("article.txt", encoding="utf-8").read(),
    summary=open("summary.txt", encoding="utf-8").read(),
)

print(result["summary_score"])
```

If the source document already has sections, pass them directly:

```python
result = metric.score(
    source_doc="",
    summary=summary,
    sections=[
        {"name": "Introduction", "text": intro_text},
        {"name": "Methods", "text": methods_text},
        {"name": "Results", "text": results_text},
    ],
)
```

The CLI also accepts sectioned input:

```bash
python -m graph_metric.score \
  --source article.txt \
  --summary summary.txt \
  --sections-json sections.json \
  --output result.json
```

`sections.json` can be either:

```json
[
  {"name": "Introduction", "text": "..."},
  {"name": "Results", "text": "..."}
]
```

or:

```json
{
  "Introduction": "...",
  "Results": "..."
}
```

## Output

The JSON output contains:

```json
{
  "metric": "GraphLongDocFACTScore",
  "summary_score": 0.74,
  "num_summary_sentences": 10,
  "num_claims": 10,
  "sentences": [],
  "claims": [],
  "graphs": []
}
```

Important fields:

- `summary_score`: final factuality score in `[0, 1]`
- `sentences`: sentence-level scores and diagnostics
- `claims`: claim-level scores and selected evidence
- `graphs`: explicit claim-evidence graph candidates with nodes, edges, feature values, and scores

## Default Metric Configuration

The canonical config is [configs/graph_metric.yaml](configs/graph_metric.yaml).

Default setup:

- claim unit: summary sentence
- retrieval: dense sentence retrieval
- embedding model: `sentence-transformers/bert-base-nli-mean-tokens`
- top-k evidence sentences per claim: `5`
- neighbor window: `0`
- entailment model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- BARTScore: disabled
- aggregation: max over evidence candidates, mean over summary sentences

Use another config with:

```bash
python -m graph_metric.score \
  --config configs/graph_metric.yaml \
  --source article.txt \
  --summary summary.txt
```

## Reproduce Benchmark Results

Dataset adapters are included only for benchmark reproduction. Normal users only need `source_doc` and `summary`.

This repository pins the public benchmark sources as git submodules for:

- LongSciVerify PubMed and ArXiv
- LongEval PubMed
- LongEval SQuALITY alignment data

They are available under `externals/LongDocFACTScore/data` and `externals/longeval-summarization/data` after running `git submodule update --init --recursive`. FENICE and ScreenEval require their own dataset preparation paths because they are not bundled in this release.

The same metric config is used for every dataset:

```bash
METRIC_CONFIG=configs/graph_metric.yaml
```

LongSciVerify PubMed:

```bash
python -m graph_metric.run_dataset_eval \
  --metric-config "$METRIC_CONFIG" \
  --dataset longsciverify \
  --subset pubmed \
  --data-root externals/LongDocFACTScore \
  --output-dir data/outputs/final_metric/longsciverify_pubmed
```

LongSciVerify ArXiv:

```bash
python -m graph_metric.run_dataset_eval \
  --metric-config "$METRIC_CONFIG" \
  --dataset longsciverify \
  --subset arxiv \
  --data-root externals/LongDocFACTScore \
  --output-dir data/outputs/final_metric/longsciverify_arxiv
```

LongEval PubMed:

```bash
python -m graph_metric.run_dataset_eval \
  --metric-config "$METRIC_CONFIG" \
  --dataset longeval \
  --subset pubmed \
  --data-root externals/LongDocFACTScore \
  --output-dir data/outputs/final_metric/longeval_pubmed
```

FENICE Story-SummEval:

```bash
python -m graph_metric.run_dataset_eval \
  --metric-config "$METRIC_CONFIG" \
  --dataset fenice \
  --subset story_summeval \
  --split test \
  --data-root externals/FENICE \
  --evaluation-task binary \
  --tune-threshold-on-validation \
  --output-dir data/outputs/final_metric/fenice_story_summeval
```

ScreenEval:

```bash
python -m graph_metric.run_dataset_eval \
  --metric-config "$METRIC_CONFIG" \
  --dataset screen_eval \
  --subset dialogue \
  --split all \
  --data-root externals/ScreenEval \
  --evaluation-task binary \
  --output-dir data/outputs/final_metric/screen_eval
```

## Runtime Benchmark

Paper-style random-15 PubMed speed test:

```bash
python graph_metric/benchmark_speed.py \
  --config configs/graph_metric.yaml \
  --dataset longsciverify \
  --subset pubmed \
  --data-root externals/LongDocFACTScore \
  --limit 15 \
  --sample-mode random \
  --seed 0 \
  --repeat 3 \
  --output-path data/outputs/final_metric/speed_pubmed_random15_seed0_k5.json
```

## Graph Diagnostics

Each candidate graph stores:

- claim node
- evidence sentence nodes
- source section node
- retrieval edge
- section edge
- optional numeric-conflict or polarity-tension edges
- feature scores used by the metric

Sentence diagnostics include labels such as:

- `strong_support`
- `mixed_support`
- `weak_support`
- `missing_anchor`
- `weak_entailment`
- `topical_not_factual`
- `numeric_conflict`
- `polarity_tension`
- `fragment_sentence`

These diagnostics are intended for error analysis: they identify which summary sentences drive a high or low factuality score and which retrieved source evidence was used.
