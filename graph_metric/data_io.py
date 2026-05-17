from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_COLUMNS = (
    ("model_1", "gencomparesumabs", "GenCompareSum-abs"),
    ("model_2", "dancersum", "DANCERSumm"),
    ("model_3", "section_conditional", "ZONESumm"),
)

LONGEVAL_SUMMARY_COLUMNS = (
    ("bigbird_pegasus", "bigbird_pegasus", "BigBird-Pegasus"),
    ("longt5", "longt5", "LongT5"),
)


@dataclass(frozen=True)
class SourceSentence:
    sentence_id: str
    document_id: str
    section_id: str
    section_name: str
    section_index: int
    sentence_index: int
    section_sentence_index: int
    text: str


@dataclass(frozen=True)
class SummaryExample:
    document_id: str
    summary_model: str
    summary_model_name: str
    summary_text: str
    summary_sentences: list[str]
    human_summary_score: float
    source_sentences: list[SourceSentence]
    abstract_text: str


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_examples(root: str | Path, dataset: str, subset: str, split: str | None = None) -> list[SummaryExample]:
    dataset = dataset.lower()
    subset = subset.lower()
    if dataset == "longsciverify":
        return load_longsciverify(root, subset)
    if dataset == "longeval" and subset == "pubmed":
        return load_longeval_pubmed(root)
    if dataset == "longeval" and subset == "squality":
        return load_longeval_squality(root)
    if dataset == "fenice" and subset == "story_summeval":
        return load_fenice_story_summeval(root, split or "test")
    if dataset == "screen_eval" and subset == "dialogue":
        return load_screen_eval(root, split or "all")
    raise ValueError(f"Unsupported dataset/subset: {dataset}/{subset}")


def load_longsciverify_pubmed(root: str | Path) -> list[SummaryExample]:
    return load_longsciverify(root, "pubmed")


def load_longsciverify(root: str | Path, subset: str) -> list[SummaryExample]:
    root = Path(root)
    subset = subset.lower()
    if subset not in {"pubmed", "arxiv"}:
        raise ValueError("LongSciVerify subset must be 'pubmed' or 'arxiv'")
    raw_path = root / "data" / "raw_data" / "LongSciVerify" / f"{subset}_test.json"
    human_dir = root / "data" / "human_eval_results" / "LongSciVerify"
    reviewer_paths = [human_dir / f"{subset}_reviewer_{idx}.csv" for idx in range(1, 4)]

    articles = {str(item["article_id"]).strip(): item for item in _read_json(raw_path)}
    reviewer_rows = [_read_csv(path) for path in reviewer_paths]
    reviewer_by_id = [{str(row.get("id", "")).strip(): row for row in rows} for rows in reviewer_rows]

    examples: list[SummaryExample] = []
    for row in reviewer_rows[0]:
        document_id = str(row.get("id", "")).strip()
        if not document_id:
            continue
        article = articles.get(document_id)
        if article is None:
            article = _find_article(document_id, articles)
        article_review_rows = [rows[document_id] for rows in reviewer_by_id if document_id in rows]
        source_sentences = parse_source_sentences(document_id, article)

        for model_key, summary_column, model_name in SUMMARY_COLUMNS:
            summary_text = clean_text(row.get(summary_column, ""))
            if not summary_text:
                continue
            human_score = human_summary_score(article_review_rows, model_key)
            examples.append(
                SummaryExample(
                    document_id=document_id,
                    summary_model=model_key,
                    summary_model_name=model_name,
                    summary_text=summary_text,
                    summary_sentences=sentence_split(summary_text),
                    human_summary_score=human_score,
                    source_sentences=source_sentences,
                    abstract_text=clean_text(article.get("abstract_text", "")),
                )
            )

    return examples


def load_longeval_pubmed(root: str | Path) -> list[SummaryExample]:
    root = Path(root)
    raw_path = root / "data" / "raw_data" / "LongEval" / "pubmed_summary_src_doc_data.json"
    score_path = root / "data" / "human_eval_results" / "LongEval" / "pubmed_fine_scores.json"
    summaries = _read_json(raw_path)
    scores = {str(item["story"]): item["score"] for item in _read_json(score_path)}

    article_ids = sorted(
        {
            key.removeprefix("article_").rsplit("_", 1)[0]
            for key in summaries
            if key.endswith("_human")
        },
        key=_natural_sort_key,
    )
    examples: list[SummaryExample] = []
    for article_id in article_ids:
        human_key = f"article_{article_id}_human"
        source_doc = summaries[human_key]["source_doc"]
        source_sentences = parse_longeval_source_sentences(f"article_{article_id}", source_doc)
        abstract_text = clean_text(summaries[human_key].get("summary", ""))
        for model_key, summary_suffix, model_name in LONGEVAL_SUMMARY_COLUMNS:
            story_key = f"article_{article_id}_{summary_suffix}"
            if story_key not in summaries or story_key not in scores:
                continue
            summary_text = clean_text(summaries[story_key]["summary"])
            human_score = float(mean(float(score) / 100.0 for score in scores[story_key]))
            examples.append(
                SummaryExample(
                    document_id=f"article_{article_id}",
                    summary_model=model_key,
                    summary_model_name=model_name,
                    summary_text=summary_text,
                    summary_sentences=sentence_split(summary_text),
                    human_summary_score=human_score,
                    source_sentences=source_sentences,
                    abstract_text=abstract_text,
                )
            )
    return examples


def load_longeval_squality(root: str | Path) -> list[SummaryExample]:
    root = Path(root)
    raw_path = root / "data" / "squality_alignment" / "data.json"
    items = _read_json(raw_path)

    examples: list[SummaryExample] = []
    for item in items:
        data = item.get("data", {})
        document_id = f"squality_{data.get('review_id', item.get('id'))}_{item.get('id')}"
        summary_text = clean_text(data.get("generation") or data.get("reference", ""))
        if not summary_text:
            continue
        source_sentences = parse_squality_source_sentences(document_id, str(data.get("document", "")))
        unit_score = squality_alignment_support_score(item)
        examples.append(
            SummaryExample(
                document_id=document_id,
                summary_model="reference" if not data.get("generation") else "generation",
                summary_model_name="SQuALITY alignment summary",
                summary_text=summary_text,
                summary_sentences=sentence_split(summary_text),
                human_summary_score=unit_score,
                source_sentences=source_sentences,
                abstract_text=clean_text(data.get("background", "")),
            )
        )
    return examples


def load_fenice_story_summeval(root: str | Path, split: str = "test") -> list[SummaryExample]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if split not in {"validation", "test", "all"}:
        raise ValueError("FENICE story-summeval split must be 'validation', 'test', or 'all'")

    rows = load_story_summeval_rows(root, split)
    source_texts = load_story_summeval_source_texts(root, rows)

    examples: list[SummaryExample] = []
    for row in rows:
        document_id = story_summeval_document_id(row)
        summary_text = clean_text(row.get("summary", ""))
        source_text = source_texts.get((str(row.get("source", "")).lower(), str(row.get("text_id", "")).strip()), "")
        if not summary_text or not source_text:
            continue
        source_sentences = parse_plain_source_sentences(document_id, source_text)
        examples.append(
            SummaryExample(
                document_id=document_id,
                summary_model=str(row.get("system_id", "unknown")),
                summary_model_name=f"Story-SummEval {row.get('system_id', 'unknown')}",
                summary_text=summary_text,
                summary_sentences=sentence_split(summary_text),
                human_summary_score=float(row.get("factual", row.get("label", 0))),
                source_sentences=source_sentences,
                abstract_text="",
            )
        )
    return examples


def load_story_summeval_rows(root: Path, split: str) -> list[dict[str, Any]]:
    local_dir = root / "story_summeval"
    local_jsonl = local_dir / f"{split}.jsonl"
    if local_jsonl.exists():
        return [json.loads(line) for line in local_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading FENICE Story-SummEval requires the `datasets` package, "
            "or local JSONL files under <root>/story_summeval/{validation,test}.jsonl."
        ) from exc

    if split == "all":
        dataset = concatenate_datasets(
            [
                load_dataset("Babelscape/story-summeval", split="validation"),
                load_dataset("Babelscape/story-summeval", split="test"),
            ]
        )
    else:
        dataset = load_dataset("Babelscape/story-summeval", split=split)
    rows = [dict(row) for row in dataset]
    local_dir.mkdir(parents=True, exist_ok=True)
    with local_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_story_summeval_source_texts(
    root: Path, rows: list[dict[str, Any]]
) -> dict[tuple[str, str], str]:
    cache_path = root / "story_summeval" / "source_text_cache.json"
    source_texts: dict[tuple[str, str], str] = {}
    if cache_path.exists():
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        source_texts.update({tuple(key.split("\t", 1)): value for key, value in raw_cache.items()})

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Fetching Story-SummEval source texts requires the `datasets` package.") from exc

    wanted_gutenberg = {
        str(row.get("text_id", "")).strip()
        for row in rows
        if str(row.get("source", "")).lower() == "gutenberg"
    }
    wanted_wikisource = {
        str(row.get("text_id", "")).strip()
        for row in rows
        if str(row.get("source", "")).lower() == "wikisource"
    }
    wanted_gutenberg = {
        text_id for text_id in wanted_gutenberg if ("gutenberg", text_id) not in source_texts
    }
    wanted_wikisource = {
        text_id for text_id in wanted_wikisource if ("wikisource", text_id) not in source_texts
    }

    if wanted_gutenberg:
        gutenberg = load_dataset("manu/project_gutenberg", split="en", streaming=True)
        remaining = set(wanted_gutenberg)
        for item in gutenberg:
            gutenberg_id = str(item.get("id", "")).split("-", 1)[0].strip()
            if gutenberg_id in remaining:
                source_texts[("gutenberg", gutenberg_id)] = clean_gutenberg_text(str(item.get("text", "")))
                remaining.remove(gutenberg_id)
                if not remaining:
                    break

    if wanted_wikisource:
        wikisource = load_dataset("wikimedia/wikisource", "20231201.en", split="train", streaming=True)
        remaining = set(wanted_wikisource)
        for item in wikisource:
            title = str(item.get("title", "")).strip()
            if title in remaining:
                source_texts[("wikisource", title)] = clean_text(item.get("text", ""))
                remaining.remove(title)
                if not remaining:
                    break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"\t".join(key): value for key, value in source_texts.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    return source_texts


def load_screen_eval(root: str | Path, split: str = "all") -> list[SummaryExample]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = load_screen_eval_rows(root, split)

    examples: list[SummaryExample] = []
    for idx, row in enumerate(expand_screen_eval_rows(rows)):
        summary_text = clean_text(row.get("summary_sentence", ""))
        source_text = clean_screen_eval_source(row.get("source", ""))
        if not summary_text or not source_text:
            continue
        convo_id = str(row.get("convo_id", f"convo_{idx}")).strip()
        summary_id = str(row.get("summary_id", "unknown")).strip()
        document_id = f"screen_eval_{re.sub(r'[^A-Za-z0-9]+', '_', convo_id).strip('_')}"
        source_sentences = parse_dialogue_source_sentences(document_id, source_text)
        examples.append(
            SummaryExample(
                document_id=document_id,
                summary_model=summary_id,
                summary_model_name=f"ScreenEval {summary_id}",
                summary_text=summary_text,
                summary_sentences=[summary_text],
                human_summary_score=1.0 if bool(row.get("label", False)) else 0.0,
                source_sentences=source_sentences,
                abstract_text="",
            )
        )
    return examples


def load_screen_eval_rows(root: Path, split: str) -> list[dict[str, Any]]:
    local_dir = root / "screen_eval"
    local_jsonl = local_dir / f"{split}.jsonl"
    if local_jsonl.exists():
        return [json.loads(line) for line in local_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading ScreenEval requires the `datasets` package, "
            "or local JSONL files under <root>/screen_eval/{all,train,test}.jsonl."
        ) from exc

    dataset = load_dataset("blattimer/ScreenEval")
    if isinstance(dataset, DatasetDict):
        if split == "all":
            rows = [dict(row) for split_dataset in dataset.values() for row in split_dataset]
        elif split in dataset:
            rows = [dict(row) for row in dataset[split]]
        elif "train" in dataset:
            rows = [dict(row) for row in dataset["train"]]
        else:
            rows = [dict(row) for split_dataset in dataset.values() for row in split_dataset]
    else:
        rows = [dict(row) for row in dataset]

    local_dir.mkdir(parents=True, exist_ok=True)
    with local_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def expand_screen_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        summaries = row.get("inferred_summary") or row.get("summary_sentence") or row.get("summary")
        labels = row.get("agg_label") or row.get("label")
        summary_ids = row.get("summary_id") or row.get("model") or row.get("system")
        convo_ids = row.get("convo_id") or row.get("dialogue_id") or row.get("id")
        source = row.get("convo") or row.get("original_convo") or row.get("prediction_annotated_source_doc")

        if isinstance(summaries, list):
            label_values = labels if isinstance(labels, list) else [labels] * len(summaries)
            summary_id_values = summary_ids if isinstance(summary_ids, list) else [summary_ids] * len(summaries)
            convo_id_values = convo_ids if isinstance(convo_ids, list) else [convo_ids] * len(summaries)
            for item_idx, summary in enumerate(summaries):
                expanded.append(
                    {
                        "summary_sentence": summary,
                        "label": label_values[item_idx] if item_idx < len(label_values) else False,
                        "summary_id": summary_id_values[item_idx] if item_idx < len(summary_id_values) else "unknown",
                        "convo_id": convo_id_values[item_idx] if item_idx < len(convo_id_values) else row_idx,
                        "source": source,
                    }
                )
        else:
            expanded.append(
                {
                    "summary_sentence": summaries,
                    "label": labels,
                    "summary_id": summary_ids or "unknown",
                    "convo_id": convo_ids or row_idx,
                    "source": source,
                }
            )
    return expanded


def parse_source_sentences(document_id: str, article: dict[str, Any]) -> list[SourceSentence]:
    section_names = article.get("section_names") or []
    sections = article.get("sections") or []
    source_sentences: list[SourceSentence] = []
    global_index = 0
    for section_index, raw_section in enumerate(sections):
        section_name = clean_text(
            section_names[section_index] if section_index < len(section_names) else f"section_{section_index + 1}"
        )
        section_id = f"{document_id}_sec{section_index:03d}"
        for section_sentence_index, raw_sentence in enumerate(raw_section):
            text = clean_text(raw_sentence)
            if not text:
                continue
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=section_name,
                    section_index=section_index,
                    sentence_index=global_index,
                    section_sentence_index=section_sentence_index,
                    text=text,
                )
            )
            global_index += 1

    if source_sentences:
        return source_sentences

    article_text = clean_text(article.get("article_text", ""))
    return [
        SourceSentence(
            sentence_id=f"{document_id}_src{idx:04d}",
            document_id=document_id,
            section_id=f"{document_id}_sec000",
            section_name="document",
            section_index=0,
            sentence_index=idx,
            section_sentence_index=idx,
            text=sentence,
        )
        for idx, sentence in enumerate(sentence_split(article_text))
    ]


def parse_longeval_source_sentences(document_id: str, source_doc: str) -> list[SourceSentence]:
    source_sentences: list[SourceSentence] = []
    global_index = 0
    sections = [clean_text(part) for part in re.split(r"<br\s*/?>", str(source_doc or ""), flags=re.I)]
    for section_index, section_text in enumerate(sections):
        if not section_text:
            continue
        section_id = f"{document_id}_sec{section_index:03d}"
        for section_sentence_index, sentence in enumerate(sentence_split(section_text)):
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=f"paragraph_{section_index + 1}",
                    section_index=section_index,
                    sentence_index=global_index,
                    section_sentence_index=section_sentence_index,
                    text=sentence,
                )
            )
            global_index += 1
    return source_sentences


def parse_squality_source_sentences(document_id: str, source_doc: str) -> list[SourceSentence]:
    text = re.sub(r"<br\s*/?>", "\n", source_doc or "", flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    paragraphs = [clean_text(part) for part in text.split("\n") if clean_text(part)]
    source_sentences: list[SourceSentence] = []
    global_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        section_id = f"{document_id}_para{paragraph_index:04d}"
        for paragraph_sentence_index, sentence in enumerate(sentence_split(paragraph)):
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=f"paragraph_{paragraph_index + 1}",
                    section_index=paragraph_index,
                    sentence_index=global_index,
                    section_sentence_index=paragraph_sentence_index,
                    text=sentence,
                )
            )
            global_index += 1
    return source_sentences


def parse_plain_source_sentences(document_id: str, source_doc: str) -> list[SourceSentence]:
    paragraphs = [clean_text(part) for part in re.split(r"\n\s*\n+", source_doc or "") if clean_text(part)]
    if not paragraphs:
        paragraphs = [clean_text(source_doc)]
    source_sentences: list[SourceSentence] = []
    global_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        section_id = f"{document_id}_para{paragraph_index:04d}"
        for paragraph_sentence_index, sentence in enumerate(sentence_split(paragraph)):
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=f"paragraph_{paragraph_index + 1}",
                    section_index=paragraph_index,
                    sentence_index=global_index,
                    section_sentence_index=paragraph_sentence_index,
                    text=sentence,
                )
            )
            global_index += 1
    return source_sentences


def parse_dialogue_source_sentences(document_id: str, source_doc: str) -> list[SourceSentence]:
    turns = [clean_text(part) for part in str(source_doc or "").split("\n") if clean_text(part)]
    if not turns:
        turns = [clean_text(source_doc)]
    source_sentences: list[SourceSentence] = []
    global_index = 0
    for turn_index, turn in enumerate(turns):
        section_id = f"{document_id}_turn{turn_index:04d}"
        for turn_sentence_index, sentence in enumerate(sentence_split(turn)):
            source_sentences.append(
                SourceSentence(
                    sentence_id=f"{document_id}_src{global_index:04d}",
                    document_id=document_id,
                    section_id=section_id,
                    section_name=f"turn_{turn_index + 1}",
                    section_index=turn_index,
                    sentence_index=global_index,
                    section_sentence_index=turn_sentence_index,
                    text=sentence,
                )
            )
            global_index += 1
    return source_sentences


def squality_alignment_support_score(item: dict[str, Any]) -> float:
    annotation_text = ""
    try:
        annotation_text = item["annotations"][0]["result"][0]["value"]["text"][0]
    except (KeyError, IndexError, TypeError):
        return 1.0

    units = parse_squality_info_units(annotation_text)
    if not units:
        return 1.0
    supported = sum(1 for unit in units if unit.get("support", "").strip())
    return supported / len(units)


def parse_squality_info_units(annotation_text: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n?(?=Info Unit #\d+\s*=)", annotation_text):
        block = block.strip()
        if not block.startswith("Info Unit #"):
            continue
        unit_match = re.search(
            r"Info Unit #\d+\s*=\s*(.*?)(?=\n(?:Contextualized|Span|Support)\s*=|$)",
            block,
            flags=re.S,
        )
        support_match = re.search(r"\nSupport\s*=\s*(.*)$", block, flags=re.S)
        units.append(
            {
                "unit": clean_text(unit_match.group(1) if unit_match else ""),
                "support": clean_text(support_match.group(1) if support_match else ""),
            }
        )
    return units


def human_summary_score(reviewer_rows: list[dict[str, str]], model_key: str) -> float:
    scores: list[float] = []
    for sentence_idx in range(1, 4):
        column = f"{model_key}_sent_{sentence_idx}_factuality"
        for row in reviewer_rows:
            score = normalize_score(row.get(column))
            if score is not None:
                scores.append(score)
    if not scores:
        raise ValueError(f"No human factuality score for {model_key}")
    return float(mean(scores))


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(clean_text(item) for item in value)
    text = str(value or "")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_gutenberg_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n")
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I | re.S)
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, flags=re.I | re.S)
    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def clean_screen_eval_source(value: Any) -> str:
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                speaker = clean_text(item.get("speaker") or item.get("role") or "")
                text = clean_text(item.get("text") or item.get("utterance") or item.get("content") or "")
                lines.append(clean_text(f"{speaker}: {text}" if speaker else text))
            elif isinstance(item, (list, tuple)):
                lines.append(clean_text(" ".join(clean_text(part) for part in item)))
            else:
                lines.append(clean_text(item))
        return "\n".join(line for line in lines if line)
    return re.sub(r"</?mark>", " ", clean_text(value))


def story_summeval_document_id(row: dict[str, Any]) -> str:
    source = str(row.get("source", "source")).strip()
    text_id = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("text_id", "text")).strip()).strip("_")
    return f"story_summeval_{source}_{text_id}"


def sentence_split(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(sentence) for sentence in sentences if clean_text(sentence)]


def normalize_score(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"yes", "true", "entailed", "supported"}:
        return 1.0
    if text in {"no", "false", "not entailed", "unsupported"}:
        return 0.0
    try:
        score = float(text)
    except ValueError:
        return None
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_article(document_id: str, articles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for article_id, article in articles.items():
        if document_id in article_id or article_id in document_id:
            return article
    raise KeyError(f"Could not find article {document_id!r}")


def _natural_sort_key(value: str) -> tuple[int, str]:
    if value.isdigit():
        return int(value), value
    return 10**9, value
