import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.vectorstore import (
    BgeM3Embeddings,
    DeterministicFakeEmbeddings,
    build_chroma_vectorstore,
    load_chunks_from_parsed_json,
)


def main():
    args = parse_args()
    embeddings = build_embeddings(args.embedding, args.model_cache_dir)
    chunks = load_chunks_from_parsed_json(
        args.parsed_json,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    vectorstore = build_chroma_vectorstore(
        chunks,
        persist_dir=args.persist_dir,
        embeddings=embeddings,
        collection_name=args.collection_name,
        reset_collection=not args.append,
    )

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Persisted Chroma vectorstore: {args.persist_dir}")
    print(f"Collection: {args.collection_name}")
    print(f"Stored vectors: {vectorstore.count()}")

    if args.query:
        print(f"Query: {args.query}")
        results = vectorstore.similarity_search(args.query, k=args.top_k)
        for index, document in enumerate(results, start=1):
            metadata = document.metadata
            preview = " ".join(document.page_content.split())[:180]
            print(
                "[{index}] source={source} page={page} type={block_type} "
                "section={section} chunk_id={chunk_id}".format(
                    index=index,
                    source=metadata.get("source", ""),
                    page=metadata.get("page", ""),
                    block_type=metadata.get("block_type", ""),
                    section=metadata.get("section", ""),
                    chunk_id=metadata.get("chunk_id", ""),
                )
            )
            print(f"    {preview}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Chroma vectorstore from MinerU-normalized finance JSON.",
    )
    parser.add_argument("--parsed-json", required=True, type=Path)
    parser.add_argument("--persist-dir", required=True, type=Path)
    parser.add_argument("--collection-name", default="finance_rag")
    parser.add_argument("--chunk-size", default=1000, type=int)
    parser.add_argument("--chunk-overlap", default=150, type=int)
    parser.add_argument(
        "--embedding",
        choices=["fake", "bge-m3"],
        default="fake",
        help="Use fake for structure tests or bge-m3 for local finance retrieval.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=PROJECT_ROOT / "data" / "model_cache",
        type=Path,
        help="Local model cache used by BGE-M3.",
    )
    parser.add_argument("--query")
    parser.add_argument("--top-k", default=4, type=int)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing Chroma collection instead of rebuilding it.",
    )
    return parser.parse_args()


def build_embeddings(name, model_cache_dir=None):
    if name == "fake":
        return DeterministicFakeEmbeddings()
    if name == "bge-m3":
        return BgeM3Embeddings(cache_dir=model_cache_dir)
    raise ValueError(f"Unsupported embedding backend: {name}")


if __name__ == "__main__":
    main()
