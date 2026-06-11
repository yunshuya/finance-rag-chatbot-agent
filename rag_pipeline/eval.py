from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RetrievalEvalItem:
    id: str
    question: str
    expected_pages: list[int]
    expected_sections: list[str] | None = None
    expected_answer_hint: str | None = None
    category: str = "numeric"


@dataclass
class RetrievalEvalResult:
    item_id: str
    question: str
    hit: bool
    reciprocal_rank: float
    retrieved_pages: list[int]
    top_sections: list[str]


def load_eval_dataset(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [
        RetrievalEvalItem(
            id=item["id"],
            question=item["question"],
            expected_pages=[int(page) for page in item["expected_pages"]],
            expected_sections=item.get("expected_sections"),
            expected_answer_hint=item.get("expected_answer_hint"),
            category=item.get("category", "numeric"),
        )
        for item in payload["items"]
    ]
    return payload, items


def document_pages(document):
    metadata = document.metadata or {}
    pages = set()
    page = metadata.get("page")
    if page is not None:
        pages.add(int(page))
    page_start = metadata.get("page_start", page)
    page_end = metadata.get("page_end", page)
    if page_start is not None and page_end is not None:
        start = int(page_start)
        end = int(page_end)
        if end >= start:
            pages.update(range(start, end + 1))
    return pages


def hit_at_k(documents, expected_pages, k):
    expected = {int(page) for page in expected_pages}
    for index, document in enumerate(documents[:k], start=1):
        if expected & document_pages(document):
            return True, 1.0 / index
    return False, 0.0


def evaluate_retrieval_case(item, documents, k):
    hit, reciprocal_rank = hit_at_k(documents, item.expected_pages, k)
    retrieved_pages = []
    top_sections = []
    for document in documents[:k]:
        retrieved_pages.extend(sorted(document_pages(document)))
        section = (document.metadata or {}).get("section")
        if section:
            top_sections.append(str(section))
    return RetrievalEvalResult(
        item_id=item.id,
        question=item.question,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        retrieved_pages=sorted(set(retrieved_pages)),
        top_sections=top_sections,
    )


def summarize_results(results: Iterable[RetrievalEvalResult]):
    results = list(results)
    total = len(results)
    hits = sum(1 for result in results if result.hit)
    mrr = sum(result.reciprocal_rank for result in results) / total if total else 0.0
    return {
        "total": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": mrr,
    }


def format_comparison_table(strategy_summaries, k):
    headers = ["strategy", f"Hit@{k}", "MRR", "hits/total"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for strategy, summary in strategy_summaries.items():
        lines.append(
            "| {strategy} | {hit_rate:.1%} | {mrr:.3f} | {hits}/{total} |".format(
                strategy=strategy,
                hit_rate=summary["hit_rate"],
                mrr=summary["mrr"],
                hits=summary["hits"],
                total=summary["total"],
            )
        )
    return "\n".join(lines)
