"""PDF preprocessing pipeline for the finance RAG chatbot."""

from .chunker import build_chunks
from .mineru_parser import MinerUParseError, find_content_list, run_mineru
from .normalizer import compute_file_sha256, make_doc_id, normalize_mineru_content
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
    "build_chunks",
    "compute_file_sha256",
    "find_content_list",
    "DeterministicFakeEmbeddings",
    "build_chroma_vectorstore",
    "load_chroma_vectorstore",
    "load_chunks_from_parsed_json",
    "make_doc_id",
    "normalize_mineru_content",
    "run_mineru",
]
