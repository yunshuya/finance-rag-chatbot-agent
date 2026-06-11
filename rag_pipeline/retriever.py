from copy import deepcopy
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from .router import QueryRouter
from .table_utils import (
    extract_relevant_table_facts,
    is_numeric_finance_question,
    keyword_overlap_score,
    rewrite_finance_query,
)


class FinanceHybridRetriever(BaseRetriever):
    """Finance-aware retriever with query rewrite, hybrid scoring, reranking, and refusal."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    llm: Optional[Any] = None
    reranker: Optional[Any] = None
    vector_k: int = 24
    final_k: int = 10
    rerank_candidates: int = 20
    vector_weight: float = 0.55
    keyword_weight: float = 0.15
    rerank_weight: float = 0.30
    table_boost: float = 0.12
    min_confidence_score: float = 0.22
    last_top_score: float = 0.0
    last_refused: bool = False

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Any]:
        return self._retrieve_documents(query)

    def get_relevant_documents(self, query):
        return self.invoke(query)

    def invoke(self, query, config=None, **kwargs):
        return self._retrieve_documents(query)

    def _retrieve_documents(self, query):
        self.last_refused = False
        self.last_top_score = 0.0

        query_variants = rewrite_finance_query(query)
        if self.llm is not None:
            query_variants.extend(_llm_rewrite_followup_query(self.llm, query))
            query_variants.extend(_llm_rewrite_queries(self.llm, query))
        query_variants = _dedupe_queries(query_variants)

        ranked = {}
        for variant in query_variants:
            for document, distance in _vector_search(self.vectorstore, variant, self.vector_k):
                chunk_id = document.metadata.get("chunk_id") or id(document)
                vector_score = 1.0 / (1.0 + float(distance))
                keyword_score = keyword_overlap_score(query, document.page_content)
                score = (
                    self.vector_weight * vector_score
                    + self.keyword_weight * keyword_score
                )
                if (
                    is_numeric_finance_question(query)
                    and document.metadata.get("block_type") == "table"
                ):
                    score += self.table_boost

                if chunk_id not in ranked or score > ranked[chunk_id][0]:
                    ranked[chunk_id] = (score, document)

        if not ranked:
            self.last_refused = True
            return []

        candidates = sorted(ranked.values(), key=lambda item: item[0], reverse=True)[
            : self.rerank_candidates
        ]
        candidate_docs = [document for _, document in candidates]
        rerank_scores = (
            self.reranker.score_pairs(query, candidate_docs)
            if self.reranker is not None
            else [0.0] * len(candidate_docs)
        )

        rescored = []
        for (hybrid_score, document), rerank_score in zip(candidates, rerank_scores):
            if self.reranker is not None:
                final_score = (
                    hybrid_score * (1.0 - self.rerank_weight)
                    + rerank_score * self.rerank_weight
                )
            else:
                final_score = hybrid_score
            rescored.append((final_score, document, rerank_score))

        rescored.sort(key=lambda item: item[0], reverse=True)
        self.last_top_score = rescored[0][0]
        if self.last_top_score < self.min_confidence_score:
            self.last_refused = True
            return []

        documents = []
        for final_score, document, rerank_score in rescored[: self.final_k]:
            enriched = _attach_extracted_facts(query, document)
            enriched.metadata = dict(enriched.metadata or {})
            enriched.metadata["retrieval_score"] = round(final_score, 4)
            enriched.metadata["rerank_score"] = round(float(rerank_score), 4)
            documents.append(enriched)
        return documents


def _dedupe_queries(queries):
    deduped = []
    seen = set()
    for query in queries:
        query = str(query or "").strip()
        if query and query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped


class RoutedDualIndexRetriever(BaseRetriever):
    """Query-router + dual-index retriever for finance annual reports."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dual_index: Any
    llm: Optional[Any] = None
    reranker: Optional[Any] = None
    router: Any = Field(default_factory=QueryRouter)
    vector_k: int = 24
    final_k: int = 10
    rerank_candidates: int = 20
    vector_weight: float = 0.55
    keyword_weight: float = 0.15
    rerank_weight: float = 0.30
    min_confidence_score: float = 0.22
    last_top_score: float = 0.0
    last_refused: bool = False
    last_route: Optional[Any] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Any]:
        return self._retrieve_documents(query)

    def get_relevant_documents(self, query):
        return self.invoke(query)

    def invoke(self, query, config=None, **kwargs):
        return self._retrieve_documents(query)

    def _retrieve_documents(self, query):
        self.last_refused = False
        self.last_top_score = 0.0
        route = self.router.route(query)
        self.last_route = route

        query_variants = rewrite_finance_query(query)
        if self.llm is not None:
            query_variants.extend(_llm_rewrite_followup_query(self.llm, query))
            query_variants.extend(_llm_rewrite_queries(self.llm, query))
        query_variants = _dedupe_queries(query_variants)

        table_k = max(4, int(self.vector_k * route.table_weight))
        text_k = max(4, int(self.vector_k * route.text_weight))

        ranked = {}
        for variant in query_variants:
            self._collect_candidates(
                ranked,
                query,
                variant,
                self.dual_index.table_store,
                table_k,
                route.table_weight,
                index_name="table",
            )
            self._collect_candidates(
                ranked,
                query,
                variant,
                self.dual_index.text_store,
                text_k,
                route.text_weight,
                index_name="text",
            )

        self._collect_fact_candidates(ranked, query, route)
        return self._finalize_ranked(query, route, ranked)

    def _collect_fact_candidates(self, ranked, query, route):
        fact_store = getattr(self.dual_index, "fact_store", None)
        if fact_store is None or fact_store.count() == 0:
            return

        if route.intent not in {"narrative_qa", "comparison", "general"}:
            return

        fact_k = 6 if route.intent == "narrative_qa" else 4
        fact_weight = 1.15 if route.intent == "narrative_qa" else 0.85
        for document in fact_store.search(query, top_k=fact_k):
            chunk_id = document.metadata.get("chunk_id") or id(document)
            base_score = float(document.metadata.get("fact_score", 0.0))
            score = fact_weight * (0.65 + base_score)
            if chunk_id not in ranked or score > ranked[chunk_id][0]:
                ranked[chunk_id] = (score, document)

    def _collect_candidates(
        self,
        ranked,
        query,
        variant,
        vectorstore,
        k,
        index_weight,
        index_name,
    ):
        for document, distance in _vector_search(vectorstore, variant, k):
            chunk_id = document.metadata.get("chunk_id") or id(document)
            vector_score = 1.0 / (1.0 + float(distance))
            keyword_score = keyword_overlap_score(query, document.page_content)
            score = index_weight * (
                self.vector_weight * vector_score
                + self.keyword_weight * keyword_score
            )
            if index_name == "table" and is_numeric_finance_question(query):
                score += 0.08 * index_weight

            if chunk_id not in ranked or score > ranked[chunk_id][0]:
                document.metadata = dict(document.metadata or {})
                document.metadata["index_name"] = index_name
                document.metadata["route_intent"] = self.last_route.intent
                ranked[chunk_id] = (score, document)

    def _finalize_ranked(self, query, route, ranked):
        if not ranked:
            self.last_refused = True
            return []

        candidates = sorted(ranked.values(), key=lambda item: item[0], reverse=True)[
            : self.rerank_candidates
        ]
        candidate_docs = [document for _, document in candidates]
        rerank_scores = (
            self.reranker.score_pairs(query, candidate_docs)
            if self.reranker is not None
            else [0.0] * len(candidate_docs)
        )

        rescored = []
        for (hybrid_score, document), rerank_score in zip(candidates, rerank_scores):
            if self.reranker is not None:
                final_score = (
                    hybrid_score * (1.0 - self.rerank_weight)
                    + rerank_score * self.rerank_weight
                )
            else:
                final_score = hybrid_score
            rescored.append((final_score, document, rerank_score))

        rescored.sort(key=lambda item: item[0], reverse=True)
        self.last_top_score = rescored[0][0]
        if self.last_top_score < self.min_confidence_score:
            self.last_refused = True
            return []

        documents = []
        for final_score, document, rerank_score in rescored[: self.final_k]:
            enriched = _attach_extracted_facts(query, document)
            enriched.metadata = dict(enriched.metadata or {})
            enriched.metadata["retrieval_score"] = round(final_score, 4)
            enriched.metadata["rerank_score"] = round(float(rerank_score), 4)
            enriched.metadata["route_intent"] = route.intent
            enriched.metadata["route_reason"] = route.reason
            documents.append(enriched)
        return documents


def _vector_search(vectorstore, query, k):
    if hasattr(vectorstore, "similarity_search_with_score"):
        return vectorstore.similarity_search_with_score(query, k=k)
    if hasattr(vectorstore, "similarity_search_with_scores"):
        results = vectorstore.similarity_search_with_scores(query, k=k)
        return [(doc, distance) for doc, distance in results]
    documents = vectorstore.similarity_search(query, k=k)
    return [(document, 0.0) for document in documents]


def _attach_extracted_facts(query, document):
    facts = extract_relevant_table_facts(query, document.page_content)
    if not facts:
        return document

    enriched = deepcopy(document)
    enriched.page_content = f"【结构化数值线索】{facts}\n\n{document.page_content}"
    enriched.metadata = dict(document.metadata or {})
    enriched.metadata["extracted_facts"] = facts
    return enriched


def _llm_rewrite_queries(llm, query):
    try:
        prompt = (
            "你是金融财报检索助手。把用户问题改写成1条更适合检索中国A股年报的独立查询，"
            "保留年份和公司指标名称，使用年报常用表述。只输出改写后的查询。\n"
            f"用户问题: {query}"
        )
        response = llm.invoke(prompt)
        rewritten = getattr(response, "content", str(response)).strip()
        return [rewritten] if rewritten else []
    except Exception:
        return []


def _llm_rewrite_followup_query(llm, query):
    if len(str(query).strip()) > 12:
        return []
    try:
        prompt = (
            "你是金融财报追问改写助手。当前问题可能是追问，请结合常见财报指标把它改写成"
            "可独立检索的完整查询。保留中文，只输出1条查询。\n"
            f"追问: {query}"
        )
        response = llm.invoke(prompt)
        rewritten = getattr(response, "content", str(response)).strip()
        return [rewritten] if rewritten else []
    except Exception:
        return []
