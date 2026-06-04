import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.chunker import build_chunks
from rag_pipeline.mineru_parser import run_mineru
from rag_pipeline.normalizer import (
    compute_file_sha256,
    make_doc_id,
    normalize_mineru_content,
    write_normalized_json,
)
from rag_pipeline.vectorstore import (
    BgeM3Embeddings,
    DeterministicFakeEmbeddings,
    build_chroma_vectorstore,
)


def collect_pdf_paths(inputs):
    """Resolve PDF inputs from files or directories, rejecting invalid entries."""
    pdf_paths = []
    seen = set()
    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            candidates = sorted(
                (item for item in path.rglob("*") if item.suffix.lower() == ".pdf"),
                key=lambda item: str(item).casefold(),
            )
        elif path.is_file() and path.suffix.lower() == ".pdf":
            candidates = [path]
        else:
            raise ValueError(f"Input is not a PDF file or directory: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            pdf_paths.append(candidate)

    if not pdf_paths:
        raise ValueError("No PDF files were found in the provided inputs.")
    return pdf_paths


def ingest_pdfs(
    inputs,
    mineru_output_dir,
    parsed_json_dir,
    vectorstore_dir,
    manifest_path,
    collection_name="finance_rag",
    embeddings=None,
    embedding_name="fake",
    model_cache_dir=None,
    run_mineru_func=run_mineru,
    reset_collection=True,
):
    """Parse PDF files, normalize JSON, chunk, vectorize, and write a manifest."""
    pdf_paths = collect_pdf_paths(inputs)
    mineru_output_dir = Path(mineru_output_dir)
    parsed_json_dir = Path(parsed_json_dir)
    vectorstore_dir = Path(vectorstore_dir)
    manifest_path = Path(manifest_path)
    mineru_output_dir.mkdir(parents=True, exist_ok=True)
    parsed_json_dir.mkdir(parents=True, exist_ok=True)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = embeddings or build_embeddings(embedding_name, model_cache_dir)
    parser_version = _package_version("mineru")
    documents = []
    stored_vectors = 0
    reset_next_success = reset_collection

    for pdf_path in pdf_paths:
        file_hash = compute_file_sha256(pdf_path)
        doc_id = make_doc_id(pdf_path.name, file_hash)
        document_record = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "source_path": str(pdf_path),
            "file_hash": file_hash,
            "status": "pending",
        }

        try:
            doc_output_dir = mineru_output_dir / doc_id
            content_list_path = run_mineru_func(
                pdf_path=pdf_path,
                output_root=doc_output_dir,
                backend="pipeline",
            )
            parsed_doc = normalize_mineru_content(
                content_list_path=content_list_path,
                filename=pdf_path.name,
                doc_id=doc_id,
                file_hash=file_hash,
                source_path=str(pdf_path),
                parser_version=parser_version,
                parse_status="success",
            )
            parsed_json_path = write_normalized_json(
                parsed_doc,
                parsed_json_dir / f"{doc_id}.json",
            )
            chunks = build_chunks(parsed_doc)
            if chunks:
                vectorstore = build_chroma_vectorstore(
                    chunks,
                    persist_dir=vectorstore_dir,
                    embeddings=embeddings,
                    collection_name=collection_name,
                    reset_collection=reset_next_success,
                )
                stored_vectors = vectorstore.count()
                reset_next_success = False

            document_record.update(
                {
                    "status": "success",
                    "content_list_path": str(content_list_path),
                    "parsed_json_path": str(parsed_json_path),
                    "mineru_output_dir": str(doc_output_dir),
                    "page_count": parsed_doc.get("page_count", 0),
                    "content_page_count": parsed_doc.get("content_page_count", 0),
                    "block_counts": parsed_doc.get("block_counts", {}),
                    "chunk_count": len(chunks),
                    "table_count": parsed_doc.get("block_counts", {}).get("table", 0),
                    "image_count": sum(
                        parsed_doc.get("block_counts", {}).get(block_type, 0)
                        for block_type in ("image", "figure", "chart")
                    ),
                }
            )
        except Exception as error:
            document_record.update(
                {
                    "status": "failed",
                    "error_message": str(error),
                    "chunk_count": 0,
                    "block_counts": {},
                }
            )

        documents.append(document_record)

    manifest = _build_manifest(
        documents=documents,
        collection_name=collection_name,
        vectorstore_dir=vectorstore_dir,
        manifest_path=manifest_path,
        embedding_name=embedding_name,
        parser_version=parser_version,
        stored_vectors=stored_vectors,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_embeddings(name, model_cache_dir=None):
    if name == "fake":
        return DeterministicFakeEmbeddings()
    if name == "bge-m3":
        return BgeM3Embeddings(
            cache_dir=model_cache_dir or PROJECT_ROOT / "data" / "model_cache"
        )
    raise ValueError(f"Unsupported embedding backend: {name}")


def _build_manifest(
    documents,
    collection_name,
    vectorstore_dir,
    manifest_path,
    embedding_name,
    parser_version,
    stored_vectors,
):
    succeeded = sum(1 for document in documents if document["status"] == "success")
    failed = sum(1 for document in documents if document["status"] == "failed")
    chunk_count = sum(document.get("chunk_count", 0) for document in documents)
    return {
        "pipeline": "finance_pdf_ingest",
        "collection_name": collection_name,
        "vectorstore_dir": str(vectorstore_dir),
        "manifest_path": str(manifest_path),
        "embedding": embedding_name,
        "parser": "mineru",
        "parser_version": parser_version,
        "summary": {
            "total_files": len(documents),
            "succeeded": succeeded,
            "failed": failed,
            "chunk_count": chunk_count,
            "stored_vectors": stored_vectors,
        },
        "documents": documents,
    }


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse finance PDFs with MinerU and store chunks in Chroma.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or directories.")
    parser.add_argument(
        "--mineru-output-dir",
        default=PROJECT_ROOT / "data" / "mineru_outputs",
        type=Path,
    )
    parser.add_argument(
        "--parsed-json-dir",
        default=PROJECT_ROOT / "data" / "parsed_json",
        type=Path,
    )
    parser.add_argument(
        "--vectorstore-dir",
        default=PROJECT_ROOT / "data" / "vector_stores" / "finance_multi_bge_m3",
        type=Path,
    )
    parser.add_argument(
        "--manifest-path",
        default=PROJECT_ROOT / "data" / "parsed_json" / "manifest.json",
        type=Path,
    )
    parser.add_argument("--collection-name", default="finance_rag")
    parser.add_argument(
        "--embedding",
        choices=["fake", "bge-m3"],
        default="bge-m3",
        help="Use fake for fast structure tests or bge-m3 for real retrieval.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=PROJECT_ROOT / "data" / "model_cache",
        type=Path,
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing Chroma collection instead of rebuilding it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = ingest_pdfs(
        inputs=args.inputs,
        mineru_output_dir=args.mineru_output_dir,
        parsed_json_dir=args.parsed_json_dir,
        vectorstore_dir=args.vectorstore_dir,
        manifest_path=args.manifest_path,
        collection_name=args.collection_name,
        embedding_name=args.embedding,
        model_cache_dir=args.model_cache_dir,
        reset_collection=not args.append,
    )
    summary = manifest["summary"]
    print(f"Processed PDFs: {summary['total_files']}")
    print(f"Succeeded: {summary['succeeded']}")
    print(f"Failed: {summary['failed']}")
    print(f"Chunks: {summary['chunk_count']}")
    print(f"Stored vectors: {summary['stored_vectors']}")
    print(f"Manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
