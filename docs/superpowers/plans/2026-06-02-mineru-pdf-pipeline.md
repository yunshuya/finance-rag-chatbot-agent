# MinerU PDF Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible PDF preprocessing pipeline that parses PDFs with MinerU, normalizes parser output into project JSON, builds retrieval chunks, and feeds those chunks into the existing Chroma vectorstore flow.

**Architecture:** Keep the Streamlit app mostly unchanged and move PDF preprocessing into a focused `rag_pipeline/` package. `RAG_app.py` will call the package for PDFs, while non-PDF loaders continue using existing LangChain loaders.

**Tech Stack:** Python, Streamlit, LangChain `Document`, Chroma, MinerU CLI, standard-library `json`, `pathlib`, `subprocess`.

---

### File Structure

- Create `rag_pipeline/__init__.py`: package marker and public imports.
- Create `rag_pipeline/mineru_parser.py`: run the MinerU CLI and locate the generated `content_list.json`.
- Create `rag_pipeline/normalizer.py`: convert MinerU content blocks into a stable project JSON document.
- Create `rag_pipeline/chunker.py`: convert normalized JSON blocks into LangChain `Document` chunks with source metadata.
- Create `tests/test_mineru_pipeline.py`: unit tests for MinerU output discovery, normalization, and chunking.
- Modify `RAG_app.py`: replace `PyPDFLoader` PDF loading with MinerU-based loading and fallback to `PyPDFLoader`.
- Modify `requirements.txt`: add `mineru[all]` and document the built-in `unittest` command for tests.
- Modify `README.md`: document MinerU setup, data pipeline, AI-assistance note, and test command.

### Task 1: Unit Tests

- [ ] Add tests that define expected behavior for MinerU output discovery.
- [ ] Add tests that define expected JSON normalization for text, table, equation, and skipped footer/header blocks.
- [ ] Add tests that define expected chunk metadata and table isolation.
- [ ] Run tests and confirm they fail before production code is added.

### Task 2: Pipeline Modules

- [ ] Implement the MinerU CLI wrapper with a clear `MinerUParseError`.
- [ ] Implement project JSON normalization.
- [ ] Implement structure-aware chunking.
- [ ] Run unit tests and confirm they pass.

### Task 3: Streamlit Integration

- [ ] Import the new pipeline in `RAG_app.py`.
- [ ] Add `load_pdf_documents_with_mineru()` and a `PyPDFLoader` fallback.
- [ ] Keep non-PDF loaders unchanged.
- [ ] Avoid double splitting MinerU chunks by marking pre-chunked documents.

### Task 4: Documentation

- [ ] Update README installation steps for MinerU.
- [ ] Explain the data flow: PDF -> MinerU -> JSON -> chunks -> Chroma.
- [ ] Document robustness behavior and course-project ownership.

### Task 5: Verification

- [ ] Run the unit test suite.
- [ ] Run Python syntax checks for the app and pipeline modules.
- [ ] Inspect `git diff` to ensure only intended files changed.
