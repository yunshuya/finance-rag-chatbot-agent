import json
import re
from pathlib import Path

from .chunker import Document
from .table_utils import keyword_overlap_score

FACT_STORE_FILENAME = "fact_store.json"

CATEGORY_PATTERNS = {
    "buyback": re.compile(r"回购"),
    "dividend": re.compile(r"分红|派息|利润分配|现金红利"),
    "risk": re.compile(r"风险|风险提示"),
    "governance": re.compile(r"治理|市值管理|投资者关系"),
    "rd_investment": re.compile(r"研发|研发投入|技术创新"),
    "business": re.compile(r"经营情况|主要经营|发展战略|市场形势"),
    "policy": re.compile(r"规划|预案|承诺|用途|注销|注册资本"),
}

QUERY_CATEGORY_HINTS = {
    "buyback": ("回购", "注销", "注册资本"),
    "dividend": ("分红", "派息", "利润分配", "现金红利"),
    "risk": ("风险",),
    "governance": ("治理", "市值管理"),
    "rd_investment": ("研发", "研发投入"),
    "business": ("经营", "发展"),
    "policy": ("规划", "预案", "计划", "用途"),
}


def _split_policy_clauses(text):
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"[。；！？\n]", text)
    clauses = []
    for part in parts:
        part = part.strip()
        if len(part) >= 12:
            clauses.append(part)
    return clauses


def _classify_clause(clause):
    categories = [
        name for name, pattern in CATEGORY_PATTERNS.items() if pattern.search(clause)
    ]
    return categories


def extract_facts_from_parsed_doc(parsed_doc):
    """Extract lightweight policy/narrative facts from normalized MinerU JSON."""
    facts = []
    doc_id = parsed_doc.get("doc_id", "document")
    filename = parsed_doc.get("filename", "")
    fact_counter = 0

    for block in parsed_doc.get("blocks", []):
        if block.get("type") != "text":
            continue
        text = str(block.get("text", "")).strip()
        if not text:
            continue

        for clause in _split_policy_clauses(text):
            categories = _classify_clause(clause)
            if not categories:
                continue
            fact_counter += 1
            facts.append(
                {
                    "fact_id": f"{doc_id}_fact_{fact_counter}",
                    "doc_id": doc_id,
                    "source": filename,
                    "text": clause,
                    "page_no": block.get("page_no"),
                    "section": block.get("section"),
                    "block_id": block.get("block_id"),
                    "categories": categories,
                }
            )
    return facts


def extract_facts_from_chunks(chunks):
    """Build policy facts from retrieval chunks when parsed JSON is unavailable."""
    if not chunks:
        return []
    sample = chunks[0].metadata or {}
    parsed_doc = {
        "doc_id": sample.get("doc_id", "document"),
        "filename": sample.get("source", ""),
        "blocks": [],
    }
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if metadata.get("block_type") not in (None, "text"):
            continue
        parsed_doc["blocks"].append(
            {
                "block_id": metadata.get("block_id", metadata.get("chunk_id")),
                "type": "text",
                "text": chunk.page_content,
                "page_no": metadata.get("page"),
                "section": metadata.get("section"),
            }
        )
    return extract_facts_from_parsed_doc(parsed_doc)


def build_fact_store(parsed_docs=None, chunks=None):
    facts = []
    for parsed_doc in parsed_docs or []:
        facts.extend(extract_facts_from_parsed_doc(parsed_doc))
    if not facts and chunks:
        facts.extend(extract_facts_from_chunks(chunks))
    return FactStore(facts)


def save_fact_store(fact_store, persist_dir):
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "fact_count": len(fact_store.facts),
        "facts": fact_store.facts,
    }
    (persist_dir / FACT_STORE_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return persist_dir / FACT_STORE_FILENAME


def load_fact_store(persist_dir):
    persist_dir = Path(persist_dir)
    path = persist_dir / FACT_STORE_FILENAME
    if not path.exists():
        return FactStore([])
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FactStore(payload.get("facts", []))


def has_fact_store(persist_dir):
    return (Path(persist_dir) / FACT_STORE_FILENAME).exists()


class FactStore:
    """In-memory keyword-searchable store for pre-extracted policy facts."""

    def __init__(self, facts):
        self.facts = list(facts or [])

    def count(self):
        return len(self.facts)

    def search(self, query, top_k=5):
        query = str(query or "").strip()
        if not query or not self.facts:
            return []

        query_categories = _infer_query_categories(query)
        scored = []
        for fact in self.facts:
            text = fact.get("text", "")
            score = keyword_overlap_score(query, text)
            fact_categories = set(fact.get("categories", []))
            if query_categories & fact_categories:
                score += 0.35
            if any(hint in query for hint in _category_hints(query_categories)):
                score += 0.10
            if score <= 0:
                continue
            scored.append((score, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._fact_to_document(score, fact) for score, fact in scored[:top_k]
        ]

    def _fact_to_document(self, score, fact):
        section = fact.get("section") or ""
        page = fact.get("page_no")
        content = f"【政策事实】{fact.get('text', '')}"
        if section:
            content = f"【章节】{section}\n{content}"
        return Document(
            page_content=content,
            metadata={
                "doc_id": fact.get("doc_id"),
                "source": fact.get("source"),
                "page": page,
                "page_start": page,
                "page_end": page,
                "section": section,
                "block_type": "policy_fact",
                "index_name": "fact_store",
                "fact_id": fact.get("fact_id"),
                "block_id": fact.get("block_id"),
                "categories": ",".join(fact.get("categories", [])),
                "fact_score": round(float(score), 4),
                "chunk_id": fact.get("fact_id"),
                "pre_chunked": True,
            },
        )


def _infer_query_categories(query):
    categories = set()
    for name, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(query):
            categories.add(name)
    for name, hints in QUERY_CATEGORY_HINTS.items():
        if any(hint in query for hint in hints):
            categories.add(name)
    return categories


def _category_hints(categories):
    hints = []
    for category in categories:
        hints.extend(QUERY_CATEGORY_HINTS.get(category, ()))
    return hints
