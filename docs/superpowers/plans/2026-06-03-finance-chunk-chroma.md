# Finance Chunking And Chroma Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible finance-report chunking and Chroma ingestion path for MinerU-normalized annual reports.

**Architecture:** Keep parsing, normalization, chunking, and vectorstore ingestion as separate steps. Text blocks are grouped by section and size; tables and images remain independent retrieval units with source metadata. Chroma ingestion accepts a parsed JSON file and an embedding backend, with deterministic fake embeddings for tests.

**Tech Stack:** Python, LangChain `Document`, Chroma, MinerU normalized JSON, `unittest`.

---

### Task 1: Finance Chunk Strategy

**Files:**
- Modify: `rag_pipeline/chunker.py`
- Test: `tests/test_mineru_pipeline.py`

- [x] Write tests that confirm text chunks flush when sections change.
- [x] Keep `table`, `image`, `chart`, `figure`, and equations as standalone chunks.
- [x] Add metadata fields needed for trustworthy retrieval: `block_id`, `section`, `asset_path`, `page_start`, and `page_end`.
- [x] Run chunker tests.

### Task 2: Chroma Ingestion Module

**Files:**
- Create: `rag_pipeline/vectorstore.py`
- Modify: `rag_pipeline/__init__.py`
- Test: `tests/test_mineru_pipeline.py`

- [x] Add deterministic fake embeddings for local tests.
- [x] Add `load_chunks_from_parsed_json`.
- [x] Add `build_chroma_vectorstore`.
- [x] Add retrieval test that verifies table chunks and metadata survive Chroma round-trip.

### Task 3: CLI Script And Real Annual Report Smoke Test

**Files:**
- Create: `scripts/build_finance_vectorstore.py`
- Modify: `README.md`

- [x] Add CLI wrapper for parsed JSON to Chroma.
- [x] Run the script against the normalized Kweichow Moutai page samples using fake embeddings.
- [x] Run a similarity query and verify retrieved metadata includes source, page, block type, and section.
