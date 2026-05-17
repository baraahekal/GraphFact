from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

from graph_metric.data_io import SourceSentence, SummaryExample


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "with", "we", "our",
}

NEGATION_TERMS = {"no", "not", "none", "without", "neither", "nor", "absence", "absent", "lack", "lacked"}
ASSOCIATION_TERMS = {"associated", "correlated", "correlation", "significant", "significantly", "increased", "decreased"}
NO_ASSOCIATION_PATTERNS = (
    r"\bno association\b",
    r"\bnot associated\b",
    r"\bno significant\b",
    r"\bwithout significant\b",
)


@dataclass(frozen=True)
class EvidenceCandidate:
    rank: int
    center_sentence_id: str
    section_id: str
    section_name: str
    retrieval_score: float
    dense_score: float
    lexical_score: float
    snippet_sentence_ids: list[str]
    snippet_text: str


class GraphMetric:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        runtime = config.get("runtime", {})
        if runtime.get("require_cuda", True) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is false")
        requested_device = runtime.get("device")
        if requested_device in {None, "auto"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        retrieval = config.get("retrieval", {})
        self.top_k = int(retrieval.get("top_k", 3))
        self.neighbor_window = int(retrieval.get("neighbor_window", 1))
        self.retrieval_mode = retrieval.get("mode", "dense")
        self.dense_weight = float(retrieval.get("dense_weight", 0.75))
        self.normalize_claim_query = bool(retrieval.get("normalize_claim_query", True))
        self.embedding_model_name = retrieval.get("embedding_model", "sentence-transformers/bert-base-nli-mean-tokens")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        scoring = config.get("scoring", {})
        self.weights = scoring.get("weights", {})
        self.sentence_aggregation = scoring.get("sentence_aggregation", "max")
        self.claim_aggregation = scoring.get("claim_aggregation", "mean")
        self.summary_aggregation = scoring.get("summary_aggregation", "sentence_mean")
        self.use_nli = bool(scoring.get("use_nli", True))
        self.nli_model_name = scoring.get("nli_model", "cross-encoder/nli-deberta-v3-large")
        self.nli_model: CrossEncoder | None = None
        if self.use_nli:
            self.nli_model = CrossEncoder(self.nli_model_name, device=self.device)

        claims_config = config.get("claim_extraction", {})
        self.claim_mode = claims_config.get("mode", "sentence")
        self.claim_extractor: LocalClaimExtractor | None = None
        if self.claim_mode == "atomic":
            self.claim_extractor = LocalClaimExtractor(claims_config, self.device)
            if claims_config.get("filter_with_nli", True) and self.nli_model is None:
                self.nli_model = CrossEncoder(
                    claims_config.get("filter_nli_model", self.nli_model_name),
                    device=self.device,
                )
        elif self.claim_mode != "sentence":
            raise ValueError("claim_extraction.mode must be 'sentence' or 'atomic'")

        self.use_bartscore = bool(scoring.get("use_bartscore", False))
        self.bartscore: BartScorer | None = None
        if self.use_bartscore:
            self.bartscore = BartScorer(
                model_name=scoring.get("bartscore_model", "facebook/bart-large"),
                device=self.device,
                batch_size=int(scoring.get("bartscore_batch_size", 8)),
                center=float(scoring.get("bartscore_center", -3.5)),
                scale=float(scoring.get("bartscore_scale", 0.7)),
            )

    def score_examples(
        self,
        examples: list[SummaryExample],
        limit: int | None = None,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        selected = examples[:limit] if limit else examples
        claim_rows: list[dict] = []
        graph_rows: list[dict] = []
        sentence_rows: list[dict] = []
        summary_rows: list[dict] = []

        progress_total = sum(len(example.summary_sentences) for example in selected)
        with tqdm(total=progress_total, desc="graph metric", unit="sent") as progress:
            for example in selected:
                example_sentence_rows: list[dict] = []
                source_texts = [sentence.text for sentence in example.source_sentences]
                source_embeddings = self.embedding_model.encode(
                    source_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).float()

                sentence_scores: list[float] = []
                claim_scores: list[float] = []
                for sent_idx, sentence in enumerate(example.summary_sentences):
                    parent_sentence_id = f"{example.document_id}_{example.summary_model}_s{sent_idx + 1:03d}"
                    claims = self.claims_for_sentence(parent_sentence_id, sentence)
                    current_sentence_scores: list[float] = []
                    current_best_graphs: list[dict] = []
                    current_claim_texts: list[str] = []
                    for claim_offset, claim in enumerate(claims):
                        claim_id = (
                            parent_sentence_id
                            if self.claim_mode == "sentence"
                            else f"{parent_sentence_id}_c{claim_offset + 1:03d}"
                        )
                        candidates = self.retrieve(claim["text"], example.source_sentences, source_embeddings)
                        graphs = self.build_candidate_graphs(example, claim_id, sent_idx, claim["text"], candidates)
                        best_graph = max(graphs, key=lambda item: item["score"]) if graphs else self.no_evidence_graph(example, claim_id, sent_idx, claim["text"])
                        claim_score = aggregate_candidate_scores(graphs, self.sentence_aggregation)
                        claim_scores.append(float(claim_score))
                        current_sentence_scores.append(float(claim_score))
                        current_best_graphs.append(best_graph)
                        current_claim_texts.append(claim["text"])

                        graph_rows.extend(graphs or [best_graph])
                        claim_rows.append(
                            {
                                "document_id": example.document_id,
                                "summary_model": example.summary_model,
                                "sentence_id": parent_sentence_id,
                                "claim_id": claim_id,
                                "claim_text": claim["text"],
                                "parent_sentence_text": sentence,
                                "claim_mode": self.claim_mode,
                                "summary_claim_entailment": claim.get("summary_entailment"),
                                "support_score": float(claim_score),
                                "best_graph_id": best_graph["graph_id"],
                                "best_graph_score": float(best_graph["score"]),
                                "best_evidence_sentence_ids": best_graph["evidence_sentence_ids"],
                                "best_section": best_graph.get("section_name", ""),
                                "sentence_aggregation": self.sentence_aggregation,
                                "features": best_graph["features"],
                            }
                        )
                    sentence_score = self.aggregate_sentence_claims(current_sentence_scores)
                    sentence_scores.append(sentence_score)
                    example_sentence_rows.append(
                        sentence_diagnostic_row(
                            example=example,
                            sentence_id=parent_sentence_id,
                            sentence_index=sent_idx,
                            sentence_text=sentence,
                            claim_texts=current_claim_texts,
                            claim_scores=current_sentence_scores,
                            best_graphs=current_best_graphs,
                            sentence_score=sentence_score,
                        )
                    )
                    progress.set_postfix_str(f"{example.document_id}_{example.summary_model}")
                    progress.update(1)

                graph_summary_score = self.aggregate_summary(sentence_scores, claim_scores)
                enrich_sentence_rows_with_summary_error(
                    example_sentence_rows,
                    human_score=example.human_summary_score,
                    graph_score=graph_summary_score,
                )
                sentence_rows.extend(example_sentence_rows)
                summary_rows.append(
                    {
                        "document_id": example.document_id,
                        "summary_model": example.summary_model,
                        "summary_model_name": example.summary_model_name,
                        "human_summary_score": example.human_summary_score,
                        "graph_summary_score": graph_summary_score,
                        "num_summary_sentences": len(example.summary_sentences),
                        "num_claims": len(claim_scores) if self.claim_mode == "atomic" else len(sentence_scores),
                    }
                )

        if self.claim_extractor is not None:
            self.claim_extractor.save_cache()
        annotate_sentence_ranks(sentence_rows)
        return claim_rows, graph_rows, sentence_rows, summary_rows

    def claims_for_sentence(self, sentence_id: str, sentence: str) -> list[dict[str, Any]]:
        if self.claim_mode == "sentence":
            return [{"text": sentence, "summary_entailment": 1.0}]
        assert self.claim_extractor is not None
        raw_claims = self.claim_extractor.extract(sentence_id, sentence)
        claims = [{"text": claim, "summary_entailment": None} for claim in raw_claims]
        if not claims:
            return [{"text": sentence, "summary_entailment": 1.0}]
        claims_config = self.config.get("claim_extraction", {})
        if not claims_config.get("filter_with_nli", True):
            return claims
        threshold = float(claims_config.get("filter_threshold", 0.55))
        scores = self.summary_claim_entailment_scores(sentence, [claim["text"] for claim in claims])
        filtered: list[dict[str, Any]] = []
        for claim, score in zip(claims, scores):
            claim["summary_entailment"] = float(score)
            if score >= threshold:
                filtered.append(claim)
        return filtered or [max(claims, key=lambda item: item["summary_entailment"] or 0.0)]

    def summary_claim_entailment_scores(self, sentence: str, claims: list[str]) -> list[float]:
        if not claims:
            return []
        if self.nli_model is None:
            return [1.0 for _ in claims]
        raw_scores = self.nli_model.predict([(sentence, claim) for claim in claims], show_progress_bar=False)
        return nli_entailment_probabilities(self.nli_model, raw_scores)

    def aggregate_sentence_claims(self, claim_scores: list[float]) -> float:
        if not claim_scores:
            return 0.0
        return aggregate_values(claim_scores, self.claim_aggregation)

    def aggregate_summary(self, sentence_scores: list[float], claim_scores: list[float]) -> float:
        if self.summary_aggregation == "claim_mean" and claim_scores:
            return float(np.mean(claim_scores))
        return float(np.mean(sentence_scores)) if sentence_scores else 0.0

    def retrieve(
        self,
        claim_text: str,
        source_sentences: list[SourceSentence],
        source_embeddings: torch.Tensor,
    ) -> list[EvidenceCandidate]:
        retrieval_claim = normalize_claim_for_retrieval(claim_text) if self.normalize_claim_query else claim_text
        claim_embedding = self.embedding_model.encode(
            retrieval_claim,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        claim_embedding = claim_embedding.reshape(1, -1).to(dtype=torch.float32)
        source_embeddings = source_embeddings.to(dtype=torch.float32)
        claim_embedding = torch.nn.functional.normalize(claim_embedding, p=2, dim=1)
        source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
        dense_scores = torch.mm(claim_embedding, source_embeddings.transpose(0, 1))[0].detach().cpu().numpy()
        lexical_scores = np.array([lexical_retrieval_score(retrieval_claim, sentence.text) for sentence in source_sentences])
        if self.retrieval_mode == "lexical":
            scores = lexical_scores
        elif self.retrieval_mode == "hybrid":
            scores = self.dense_weight * normalize_array(dense_scores) + (1.0 - self.dense_weight) * normalize_array(lexical_scores)
        else:
            scores = dense_scores
        ranked_indices = np.argsort(-scores)

        candidates: list[EvidenceCandidate] = []
        used_centers: set[int] = set()
        for source_index in ranked_indices:
            source_index = int(source_index)
            if source_index in used_centers:
                continue
            used_centers.add(source_index)
            center = source_sentences[source_index]
            snippet_sentences = surrounding_sentences(source_sentences, source_index, self.neighbor_window)
            candidates.append(
                EvidenceCandidate(
                    rank=len(candidates) + 1,
                    center_sentence_id=center.sentence_id,
                    section_id=center.section_id,
                    section_name=center.section_name,
                    retrieval_score=float(scores[source_index]),
                    dense_score=float(dense_scores[source_index]),
                    lexical_score=float(lexical_scores[source_index]),
                    snippet_sentence_ids=[sentence.sentence_id for sentence in snippet_sentences],
                    snippet_text=" ".join(sentence.text for sentence in snippet_sentences),
                )
            )
            if len(candidates) >= self.top_k:
                break
        return candidates

    def build_candidate_graphs(
        self,
        example: SummaryExample,
        claim_id: str,
        sentence_index: int,
        claim_text: str,
        candidates: list[EvidenceCandidate],
    ) -> list[dict]:
        nli_scores = self.nli_scores(claim_text, [candidate.snippet_text for candidate in candidates])
        bart_scores = self.bart_scores(claim_text, [candidate.snippet_text for candidate in candidates])
        graphs: list[dict] = []
        for candidate, nli_score, bart_score in zip(candidates, nli_scores, bart_scores):
            features = graph_features(claim_text, candidate, float(nli_score), float(bart_score))
            score = combine_features(features, self.weights)
            graph_id = stable_id([claim_id, candidate.center_sentence_id, candidate.snippet_text])
            graphs.append(
                {
                    "graph_id": graph_id,
                    "document_id": example.document_id,
                    "summary_model": example.summary_model,
                    "sentence_index": sentence_index,
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "candidate_rank": candidate.rank,
                    "section_id": candidate.section_id,
                    "section_name": candidate.section_name,
                    "evidence_sentence_ids": candidate.snippet_sentence_ids,
                    "evidence_text": candidate.snippet_text,
                    "features": features,
                    "score": score,
                    "nodes": graph_nodes(claim_id, claim_text, candidate),
                    "edges": graph_edges(claim_id, candidate, features),
                }
            )
        return graphs

    def nli_scores(self, claim_text: str, snippets: list[str]) -> list[float]:
        if not snippets:
            return []
        if self.nli_model is None:
            return [0.5 for _ in snippets]
        raw_scores = self.nli_model.predict([(snippet, claim_text) for snippet in snippets], show_progress_bar=False)
        return nli_entailment_probabilities(self.nli_model, raw_scores)

    def bart_scores(self, claim_text: str, snippets: list[str]) -> list[float]:
        if not snippets:
            return []
        if self.bartscore is None:
            return [0.5 for _ in snippets]
        return self.bartscore.score(snippets, [claim_text for _ in snippets])

    def no_evidence_graph(self, example: SummaryExample, claim_id: str, sentence_index: int, claim_text: str) -> dict:
        graph_id = stable_id([claim_id, "no_evidence"])
        return {
            "graph_id": graph_id,
            "document_id": example.document_id,
            "summary_model": example.summary_model,
            "sentence_index": sentence_index,
            "claim_id": claim_id,
            "claim_text": claim_text,
            "candidate_rank": None,
            "section_id": None,
            "section_name": "",
            "evidence_sentence_ids": [],
            "evidence_text": "",
            "features": {"no_evidence": 1.0},
            "score": 0.0,
            "nodes": [{"id": claim_id, "type": "claim", "text": claim_text}],
            "edges": [],
        }


def surrounding_sentences(sentences: list[SourceSentence], center_index: int, window: int) -> list[SourceSentence]:
    center = sentences[center_index]
    selected: list[SourceSentence] = []
    for idx in range(max(0, center_index - window), min(len(sentences), center_index + window + 1)):
        if sentences[idx].section_id == center.section_id:
            selected.append(sentences[idx])
    return selected


def graph_features(claim: str, candidate: EvidenceCandidate, nli_score: float, bartscore: float = 0.5) -> dict[str, float]:
    retrieval_claim = normalize_claim_for_retrieval(claim)
    claim_tokens = content_tokens(retrieval_claim)
    evidence_tokens = content_tokens(candidate.snippet_text)
    claim_values = extract_values(retrieval_claim)
    evidence_values = extract_values(candidate.snippet_text)

    lexical_anchor_coverage = overlap_ratio(claim_tokens, evidence_tokens)
    evidence_precision = overlap_ratio(evidence_tokens, claim_tokens)
    numeric_match, numeric_mismatch = numeric_consistency(claim_values, evidence_values)
    polarity = polarity_consistency(claim, candidate.snippet_text)
    section_score = section_relevance(claim, candidate.section_name)
    missing_anchors = 1.0 if claim_tokens and lexical_anchor_coverage < 0.35 else 0.0

    return {
        "retrieval_score": clamp01((candidate.retrieval_score + 1.0) / 2.0),
        "dense_retrieval_score": clamp01((candidate.dense_score + 1.0) / 2.0),
        "lexical_retrieval_score": candidate.lexical_score,
        "nli_support": clamp01(nli_score),
        "bartscore_support": clamp01(bartscore),
        "lexical_anchor_coverage": lexical_anchor_coverage,
        "evidence_precision": evidence_precision,
        "numeric_match": numeric_match,
        "numeric_mismatch": numeric_mismatch,
        "polarity_consistency": polarity,
        "section_relevance": section_score,
        "missing_anchor_penalty": missing_anchors,
    }


def combine_features(features: dict[str, float], weights: dict[str, float]) -> float:
    default_weights = {
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
    }
    merged = {**default_weights, **weights}
    score = sum(float(weight) * float(features.get(name, 0.0)) for name, weight in merged.items())
    return clamp01(score)


def aggregate_candidate_scores(graphs: list[dict], mode: str) -> float:
    if not graphs:
        return 0.0
    scores = np.array([float(graph["score"]) for graph in graphs])
    return aggregate_values(scores.tolist(), mode, ranks=[int(graph["candidate_rank"]) for graph in graphs])


def aggregate_values(values: list[float], mode: str, ranks: list[int] | None = None) -> float:
    if not values:
        return 0.0
    scores = np.array(values, dtype=float)
    if mode == "top2_mean":
        return float(np.mean(np.sort(scores)[-min(2, len(scores)) :]))
    if mode == "top3_mean":
        return float(np.mean(np.sort(scores)[-min(3, len(scores)) :]))
    if mode == "bottom2_mean":
        return float(np.mean(np.sort(scores)[: min(2, len(scores))]))
    if mode == "min":
        return float(np.min(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "rank_weighted":
        if not ranks:
            return float(np.mean(scores))
        weights = np.array([1.0 / float(rank) for rank in ranks])
        return float(np.average(scores, weights=weights))
    if mode == "softmax":
        weights = softmax(scores.reshape(1, -1) * 8.0, axis=1)[0]
        return float(np.sum(weights * scores))
    return float(np.max(scores))


def graph_nodes(claim_id: str, claim_text: str, candidate: EvidenceCandidate) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"id": claim_id, "type": "claim", "text": claim_text},
        {"id": candidate.section_id, "type": "section", "text": candidate.section_name},
    ]
    for sentence_id in candidate.snippet_sentence_ids:
        nodes.append({"id": sentence_id, "type": "evidence_sentence"})
    for idx, value in enumerate(extract_values(claim_text)):
        nodes.append({"id": f"{claim_id}_value_{idx}", "type": "claim_value", **asdict(value)})
    for idx, value in enumerate(extract_values(candidate.snippet_text)):
        nodes.append({"id": f"{candidate.center_sentence_id}_value_{idx}", "type": "evidence_value", **asdict(value)})
    return nodes


def graph_edges(claim_id: str, candidate: EvidenceCandidate, features: dict[str, float]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = [
        {
            "source": claim_id,
            "target": candidate.center_sentence_id,
            "type": "retrieved_evidence",
            "rank": candidate.rank,
            "retrieval_score": features["retrieval_score"],
            "dense_score": features["dense_retrieval_score"],
            "lexical_score": features["lexical_retrieval_score"],
        },
        {"source": candidate.center_sentence_id, "target": candidate.section_id, "type": "belongs_to_section"},
    ]
    if features["numeric_mismatch"] > 0:
        edges.append({"source": claim_id, "target": candidate.center_sentence_id, "type": "numeric_conflict"})
    if features["polarity_consistency"] < 1:
        edges.append({"source": claim_id, "target": candidate.center_sentence_id, "type": "polarity_tension"})
    return edges


@dataclass(frozen=True)
class ValueMention:
    kind: str
    text: str
    value: float | None


def extract_values(text: str) -> list[ValueMention]:
    values: list[ValueMention] = []
    for match in re.finditer(r"\bp\s*[<=>]\s*0?\.\d+\b", text, flags=re.I):
        values.append(ValueMention("p_value", match.group(0).lower().replace(" ", ""), parse_float(match.group(0))))
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*%", text):
        values.append(ValueMention("percentage", match.group(0).replace(" ", ""), parse_float(match.group(0))))
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b", text):
        values.append(ValueMention("ratio", match.group(0).replace(" ", ""), None))
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        token = match.group(0)
        if any(token in value.text for value in values):
            continue
        values.append(ValueMention("number", token, parse_float(token)))
    return values


def numeric_consistency(claim_values: list[ValueMention], evidence_values: list[ValueMention]) -> tuple[float, float]:
    if not claim_values:
        return 1.0, 0.0
    if not evidence_values:
        return 0.0, 1.0

    matched = 0
    mismatched = 0
    for claim_value in claim_values:
        same_kind = [value for value in evidence_values if value.kind == claim_value.kind]
        pool = same_kind or evidence_values
        if any(values_match(claim_value, evidence_value) for evidence_value in pool):
            matched += 1
        else:
            mismatched += 1
    total = max(len(claim_values), 1)
    return matched / total, mismatched / total


def values_match(left: ValueMention, right: ValueMention) -> bool:
    if left.text == right.text:
        return True
    if left.value is None or right.value is None:
        return False
    tolerance = 0.01 if left.kind in {"p_value", "percentage"} else max(0.1, abs(left.value) * 0.02)
    return abs(left.value - right.value) <= tolerance


def parse_float(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def polarity_consistency(claim: str, evidence: str) -> float:
    claim_no_assoc = any(re.search(pattern, claim, flags=re.I) for pattern in NO_ASSOCIATION_PATTERNS)
    evidence_no_assoc = any(re.search(pattern, evidence, flags=re.I) for pattern in NO_ASSOCIATION_PATTERNS)
    claim_assoc = bool(set(content_tokens(claim)) & ASSOCIATION_TERMS)
    evidence_assoc = bool(set(content_tokens(evidence)) & ASSOCIATION_TERMS)
    claim_neg = bool(set(content_tokens(claim)) & NEGATION_TERMS)
    evidence_neg = bool(set(content_tokens(evidence)) & NEGATION_TERMS)

    if claim_no_assoc != evidence_no_assoc and (claim_assoc or evidence_assoc):
        return 0.0
    if claim_neg != evidence_neg and lexical_core_overlap(claim, evidence) >= 0.45:
        return 0.5
    return 1.0


def section_relevance(claim: str, section_name: str) -> float:
    claim_tokens = set(content_tokens(claim))
    section = section_name.lower()
    if claim_tokens & {"method", "methods", "patients", "sample", "included", "performed", "collected"}:
        return 1.0 if any(term in section for term in ["method", "material", "patient"]) else 0.65
    if claim_tokens & {"result", "results", "found", "detected", "associated", "significant", "p"}:
        return 1.0 if "result" in section or "discussion" in section else 0.7
    if claim_tokens & {"background", "common", "worldwide", "prevalent"}:
        return 1.0 if "background" in section or "introduction" in section else 0.75
    return 0.85


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def normalize_claim_for_retrieval(text: str) -> str:
    text = re.sub(
        r"^\s*(background|objective|objectives|aim|aims|methods?|materials and methods|results?|conclusions?|introduction)\s*[:\-]\s*",
        "",
        text,
        flags=re.I,
    )
    return text.strip()


def lexical_retrieval_score(claim: str, evidence: str) -> float:
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(content_tokens(evidence))
    overlap = claim_tokens & evidence_tokens
    score = len(overlap) / len(claim_tokens)

    claim_values = {value.text for value in extract_values(claim)}
    evidence_values = {value.text for value in extract_values(evidence)}
    if claim_values:
        value_overlap = len(claim_values & evidence_values) / len(claim_values)
        score = 0.75 * score + 0.25 * value_overlap
    return clamp01(score)


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def overlap_ratio(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    return len(set(left) & set(right)) / len(set(left))


def lexical_core_overlap(left: str, right: str) -> float:
    return overlap_ratio(content_tokens(left), content_tokens(right))


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def nli_entailment_probabilities(model: CrossEncoder, raw_scores: Any) -> list[float]:
    scores = np.asarray(raw_scores)
    if scores.ndim == 1:
        return [float(sigmoid(score)) for score in scores]

    labels = [str(label).lower() for label in getattr(model.model.config, "id2label", {}).values()]
    entailment_index = next((idx for idx, label in enumerate(labels) if "entail" in label), scores.shape[-1] - 1)
    probs = softmax(scores, axis=1)
    return [float(value) for value in probs[:, entailment_index]]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stable_id(parts: list[str]) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"graph_{digest}"


def sentence_diagnostic_row(
    example: SummaryExample,
    sentence_id: str,
    sentence_index: int,
    sentence_text: str,
    claim_texts: list[str],
    claim_scores: list[float],
    best_graphs: list[dict],
    sentence_score: float,
) -> dict[str, Any]:
    if best_graphs:
        diagnostic_index = int(np.argmin([float(graph["score"]) for graph in best_graphs]))
        diagnostic_graph = best_graphs[diagnostic_index]
        diagnostic_claim = claim_texts[diagnostic_index] if diagnostic_index < len(claim_texts) else sentence_text
    else:
        diagnostic_graph = {}
        diagnostic_claim = sentence_text

    features = diagnostic_graph.get("features", {})
    row: dict[str, Any] = {
        "document_id": example.document_id,
        "summary_model": example.summary_model,
        "summary_model_name": example.summary_model_name,
        "sentence_id": sentence_id,
        "sentence_index": sentence_index + 1,
        "sentence_text": sentence_text,
        "sentence_score": float(sentence_score),
        "claim_count": len(claim_scores),
        "claim_score_min": float(min(claim_scores)) if claim_scores else 0.0,
        "claim_score_mean": float(np.mean(claim_scores)) if claim_scores else 0.0,
        "claim_score_max": float(max(claim_scores)) if claim_scores else 0.0,
        "diagnostic_claim_text": diagnostic_claim,
        "diagnostic_graph_id": diagnostic_graph.get("graph_id", ""),
        "diagnostic_graph_score": float(diagnostic_graph.get("score", 0.0)),
        "diagnostic_evidence_sentence_ids": "|".join(diagnostic_graph.get("evidence_sentence_ids", [])),
        "diagnostic_evidence_text": diagnostic_graph.get("evidence_text", ""),
        "diagnostic_section": diagnostic_graph.get("section_name", ""),
    }
    for name in [
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
        "section_relevance",
        "missing_anchor_penalty",
    ]:
        row[f"feature_{name}"] = float(features.get(name, 0.0))

    row["failure_risk"] = sentence_failure_risk(row)
    labels, explanation = sentence_diagnostic_labels(row)
    row["diagnostic_labels"] = "|".join(labels)
    row["diagnostic_explanation"] = explanation
    return row


def sentence_failure_risk(row: dict[str, Any]) -> float:
    score = float(row.get("sentence_score", 0.0))
    return clamp01(
        0.45 * (1.0 - score)
        + 0.20 * float(row.get("feature_numeric_mismatch", 0.0))
        + 0.15 * (1.0 - float(row.get("feature_polarity_consistency", 1.0)))
        + 0.10 * float(row.get("feature_missing_anchor_penalty", 0.0))
        + 0.05 * (1.0 - float(row.get("feature_lexical_anchor_coverage", 0.0)))
        + 0.05 * (1.0 - float(row.get("feature_nli_support", 0.0)))
    )


def sentence_diagnostic_labels(row: dict[str, Any]) -> tuple[list[str], str]:
    labels: list[str] = []
    reasons: list[str] = []
    sentence = str(row.get("sentence_text", ""))
    section = str(row.get("diagnostic_section", "")).lower()
    sentence_score = float(row.get("sentence_score", 0.0))
    retrieval_score = float(row.get("feature_retrieval_score", 0.0))
    nli_support = float(row.get("feature_nli_support", 0.0))
    bartscore_support = float(row.get("feature_bartscore_support", 0.0))
    anchor_coverage = float(row.get("feature_lexical_anchor_coverage", 0.0))
    numeric_mismatch = float(row.get("feature_numeric_mismatch", 0.0))
    polarity = float(row.get("feature_polarity_consistency", 1.0))
    missing_anchor = float(row.get("feature_missing_anchor_penalty", 0.0))

    if sentence_fragment(sentence):
        labels.append("fragment_sentence")
        reasons.append("summary sentence is too short or fragmentary")
    if anchor_coverage < 0.35 or missing_anchor > 0:
        labels.append("missing_anchor")
        reasons.append("important claim anchors are weak or missing in evidence")
    if nli_support < 0.20:
        labels.append("weak_entailment")
        reasons.append("evidence has low entailment support for the claim")
    if retrieval_score >= 0.75 and nli_support < 0.25:
        labels.append("topical_not_factual")
        reasons.append("retrieval is high but entailment is low")
    if numeric_mismatch > 0:
        labels.append("numeric_conflict")
        reasons.append("claim values do not match retrieved evidence values")
    if polarity < 0.75:
        labels.append("polarity_tension")
        reasons.append("negation or association polarity may differ")
    if sentence_score >= 0.60 and nli_support < 0.25:
        labels.append("high_score_weak_entailment")
        reasons.append("graph score is moderate/high despite weak entailment")
    if sentence_score >= 0.60 and bartscore_support < 0.20:
        labels.append("high_score_weak_generation_likelihood")
        reasons.append("graph score is moderate/high despite weak generation likelihood")
    if result_like_claim(sentence) and discussion_like_section(section):
        labels.append("discussion_evidence_for_result_claim")
        reasons.append("result-like claim is supported by discussion/background evidence")
    if not labels:
        if sentence_score >= 0.80:
            labels.append("strong_support")
            reasons.append("graph found strong support evidence")
        elif sentence_score <= 0.40:
            labels.append("weak_support")
            reasons.append("graph found weak support evidence")
        else:
            labels.append("mixed_support")
            reasons.append("graph found partial or mixed support")
    return labels, "; ".join(dict.fromkeys(reasons))


def sentence_fragment(sentence: str) -> bool:
    tokens = content_tokens(sentence)
    if len(tokens) <= 3:
        return True
    if len(tokens) <= 5 and not any(char.isdigit() for char in sentence):
        return True
    return False


def result_like_claim(sentence: str) -> bool:
    text = sentence.lower()
    if re.search(r"\b\d+(?:\.\d+)?\s*%|\bp\s*[<=>]\s*0?\.\d+|\bresults?\b|\bfound\b|\bdetected\b|\bobserved\b|\bsignificant", text):
        return True
    return False


def discussion_like_section(section: str) -> bool:
    return any(name in section for name in ["discussion", "background", "introduction"])


def enrich_sentence_rows_with_summary_error(sentence_rows: list[dict], human_score: float, graph_score: float) -> None:
    summary_error = float(graph_score) - float(human_score)
    if summary_error > 0.15:
        error_type = "summary_false_high"
    elif summary_error < -0.15:
        error_type = "summary_false_low"
    else:
        error_type = "summary_near_human"

    for row in sentence_rows:
        sentence_score = float(row.get("sentence_score", 0.0))
        if summary_error > 0:
            pressure = abs(summary_error) * sentence_score
        else:
            pressure = abs(summary_error) * (1.0 - sentence_score)
        row["human_summary_score"] = float(human_score)
        row["graph_summary_score"] = float(graph_score)
        row["summary_error"] = summary_error
        row["summary_abs_error"] = abs(summary_error)
        row["summary_error_type"] = error_type
        row["summary_disagreement_pressure"] = float(pressure)


def annotate_sentence_ranks(sentence_rows: list[dict]) -> None:
    for rank, row in enumerate(sorted(sentence_rows, key=lambda item: -float(item["sentence_score"])), start=1):
        row["support_rank"] = rank
    for rank, row in enumerate(sorted(sentence_rows, key=lambda item: -float(item["failure_risk"])), start=1):
        row["failure_risk_rank"] = rank
    for rank, row in enumerate(
        sorted(sentence_rows, key=lambda item: -float(item["summary_disagreement_pressure"])),
        start=1,
    ):
        row["summary_disagreement_rank"] = rank


def load_seq2seq_model(model_name: str, device: str) -> AutoModelForSeq2SeqLM:
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    except Exception as first_error:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except ValueError as second_error:
            if "torch.load" in str(second_error) and "vulnerability" in str(second_error):
                raise RuntimeError(
                    f"Could not safely load {model_name}. This environment has an older torch version, "
                    "and Transformers now requires torch>=2.6 when a model falls back to .bin weights. "
                    "Install torch>=2.6, or use a model checkpoint with safetensors weights."
                ) from second_error
            raise second_error from first_error
    return model.to(device)


class BartScorer:
    def __init__(self, model_name: str, device: str, batch_size: int, center: float, scale: float):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.center = center
        self.scale = scale
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = load_seq2seq_model(model_name, device)
        self.model.eval()
        self.loss_fct = nn.NLLLoss(reduction="none", ignore_index=self.model.config.pad_token_id)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def score(self, sources: list[str], targets: list[str]) -> list[float]:
        support_scores: list[float] = []
        for start in range(0, len(sources), self.batch_size):
            batch_sources = sources[start : start + self.batch_size]
            batch_targets = targets[start : start + self.batch_size]
            support_scores.extend(self._score_batch(batch_sources, batch_targets))
        return support_scores

    def _score_batch(self, sources: list[str], targets: list[str]) -> list[float]:
        with torch.no_grad():
            encoded_sources = self.tokenizer(
                sources,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded_targets = self.tokenizer(
                targets,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            source_ids = encoded_sources["input_ids"].to(self.device)
            source_mask = encoded_sources["attention_mask"].to(self.device)
            target_ids = encoded_targets["input_ids"].to(self.device)
            target_mask = encoded_targets["attention_mask"].to(self.device)
            target_lengths = target_mask.sum(dim=1).clamp(min=1)

            output = self.model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
            logits = output.logits.reshape(-1, self.model.config.vocab_size)
            loss = self.loss_fct(self.log_softmax(logits), target_ids.reshape(-1))
            loss = loss.reshape(target_ids.shape[0], -1)
            per_token_loss = loss.sum(dim=1) / target_lengths
            raw_bart_scores = [-float(value.item()) for value in per_token_loss]
            return [float(sigmoid((score - self.center) / self.scale)) for score in raw_bart_scores]


class LocalClaimExtractor:
    def __init__(self, config: dict[str, Any], device: str):
        self.config = config
        self.device = device
        self.model_name = config.get("model", "Babelscape/t5-base-summarization-claim-extractor")
        self.batch_size = int(config.get("batch_size", 8))
        self.max_input_length = int(config.get("max_input_length", 512))
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.num_beams = int(config.get("num_beams", 4))
        self.max_claims_per_sentence = int(config.get("max_claims_per_sentence", 8))
        self.cache_path = Path(config.get("cache_path", "data/cache/graph_metric/atomic_claims.json"))
        self.cache: dict[str, list[str]] = self.load_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = load_seq2seq_model(self.model_name, device)
        self.model.eval()

    def extract(self, sentence_id: str, sentence: str) -> list[str]:
        cache_key = stable_id([self.model_name, sentence])
        if cache_key in self.cache:
            return self.cache[cache_key]
        generated = self.generate(sentence)
        claims = parse_generated_claims(generated)
        claims = claims[: self.max_claims_per_sentence]
        if not claims and sentence.strip():
            claims = [sentence.strip()]
        self.cache[cache_key] = claims
        return claims

    def generate(self, sentence: str) -> str:
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def load_cache(self) -> dict[str, list[str]]:
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {str(key): [str(item) for item in value] for key, value in data.items()}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self.cache, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def parse_generated_claims(text: str) -> list[str]:
    text = text.replace("<pad>", " ").replace("</s>", " ")
    text = re.sub(r"<extra_id_\d+>", "\n", text)
    text = re.sub(r"\s*(?:<sep>|\[SEP\]|\|\||•)\s*", "\n", text, flags=re.I)
    text = re.sub(r"\s+\d+\.\s+", "\n", text)
    raw_parts: list[str] = []
    for part in re.split(r"[\n;]+", text):
        raw_parts.extend(split_claim_sentences(part))

    claims: list[str] = []
    for part in raw_parts:
        claim = clean_generated_claim(part)
        if len(claim) < 8:
            continue
        if is_near_duplicate_claim(claim, claims):
            continue
        claims.append(claim)
    return claims


def split_claim_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def clean_generated_claim(text: str) -> str:
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,;:")
    return text


def is_near_duplicate_claim(claim: str, existing_claims: list[str]) -> bool:
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return True
    claim_key = claim_normalized_key(claim)
    for existing in existing_claims:
        existing_key = claim_normalized_key(existing)
        if claim_key == existing_key:
            return True
        if claim_key.startswith(existing_key) or existing_key.startswith(claim_key):
            return True
        existing_tokens = set(content_tokens(existing))
        if existing_tokens:
            overlap = len(claim_tokens & existing_tokens) / len(claim_tokens | existing_tokens)
            if overlap >= 0.88:
                return True
    return False


def claim_normalized_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
