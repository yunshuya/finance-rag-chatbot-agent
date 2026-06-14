"""PDF preprocessing pipeline for the finance RAG chatbot."""

from .chunker import build_chunks
from .eval import (
    evaluate_retrieval_case,
    hit_at_k,
    load_eval_dataset,
    summarize_results,
)
from .fact_store import FactStore, build_fact_store, extract_facts_from_parsed_doc, load_fact_store
from .dual_index import (
    DualChromaIndex,
    build_dual_chroma_index,
    has_dual_chroma_index,
    load_dual_chroma_index,
    split_chunks_for_dual_index,
)
from .mineru_parser import MinerUParseError, find_content_list, run_mineru
from .normalizer import compute_file_sha256, make_doc_id, normalize_mineru_content
from .reranker import BgeReranker
from .retriever import FinanceHybridRetriever, RoutedDualIndexRetriever
from .router import QueryRouter, RouteDecision
from .table_utils import (
    enrich_table_block,
    extract_relevant_table_facts,
    html_table_to_markdown,
    is_numeric_finance_question,
    rewrite_finance_query,
)
from .vectorstore import (
    BgeM3Embeddings,
    DeterministicFakeEmbeddings,
    build_chroma_vectorstore,
    load_chroma_vectorstore,
    load_chunks_from_parsed_json,
)

__all__ = [
    "MinerUParseError",
    "BgeM3Embeddings",
    "BgeReranker",
    "DualChromaIndex",
    "FactStore",
    "build_fact_store",
    "extract_facts_from_parsed_doc",
    "load_fact_store",
    "FinanceHybridRetriever",
    "QueryRouter",
    "RouteDecision",
    "RoutedDualIndexRetriever",
    "build_chunks",
    "build_dual_chroma_index",
    "evaluate_retrieval_case",
    "hit_at_k",
    "load_eval_dataset",
    "summarize_results",
    "compute_file_sha256",
    "enrich_table_block",
    "extract_relevant_table_facts",
    "find_content_list",
    "DeterministicFakeEmbeddings",
    "build_chroma_vectorstore",
    "html_table_to_markdown",
    "is_numeric_finance_question",
    "has_dual_chroma_index",
    "load_chroma_vectorstore",
    "load_dual_chroma_index",
    "load_chunks_from_parsed_json",
    "make_doc_id",
    "normalize_mineru_content",
    "rewrite_finance_query",
    "run_mineru",
]
