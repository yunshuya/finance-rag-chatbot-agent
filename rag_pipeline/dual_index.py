import json
from pathlib import Path

from .fact_store import build_fact_store, load_fact_store, save_fact_store
from .vectorstore import build_chroma_vectorstore, load_chroma_vectorstore

TABLE_SUFFIX = "__tables"
TEXT_SUFFIX = "__text"
META_FILENAME = "dual_index_meta.json"

TABLE_BLOCK_TYPES = {"table"}


def dual_collection_names(collection_name):
    return f"{collection_name}{TABLE_SUFFIX}", f"{collection_name}{TEXT_SUFFIX}"


def split_chunks_for_dual_index(chunks):
    table_chunks = []
    text_chunks = []
    for chunk in chunks:
        block_type = (chunk.metadata or {}).get("block_type", "text")
        if block_type in TABLE_BLOCK_TYPES:
            table_chunks.append(chunk)
        else:
            text_chunks.append(chunk)
    return table_chunks, text_chunks


class DualChromaIndex:
    """Table and text Chroma collections stored under one persist directory."""

    def __init__(
        self,
        table_store,
        text_store,
        collection_name,
        persist_dir,
        fact_store=None,
    ):
        self.table_store = table_store
        self.text_store = text_store
        self.fact_store = fact_store
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)

    def count(self):
        return self.table_store.count() + self.text_store.count()

    def table_count(self):
        return self.table_store.count()

    def text_count(self):
        return self.text_store.count()


def build_dual_chroma_index(
    chunks,
    persist_dir,
    embeddings=None,
    collection_name="finance_rag",
    reset_collection=True,
    parsed_docs=None,
):
    persist_dir = Path(persist_dir)
    table_chunks, text_chunks = split_chunks_for_dual_index(chunks)
    table_name, text_name = dual_collection_names(collection_name)

    table_store = build_chroma_vectorstore(
        table_chunks,
        persist_dir=persist_dir,
        embeddings=embeddings,
        collection_name=table_name,
        reset_collection=reset_collection,
    )
    text_store = build_chroma_vectorstore(
        text_chunks,
        persist_dir=persist_dir,
        embeddings=embeddings,
        collection_name=text_name,
        reset_collection=reset_collection,
    )

    fact_store = build_fact_store(parsed_docs=parsed_docs, chunks=text_chunks)
    save_fact_store(fact_store, persist_dir)

    meta = {
        "collection_name": collection_name,
        "table_collection": table_name,
        "text_collection": text_name,
        "table_count": table_store.count(),
        "text_count": text_store.count(),
        "fact_count": fact_store.count(),
        "total_count": table_store.count() + text_store.count(),
    }
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return DualChromaIndex(
        table_store=table_store,
        text_store=text_store,
        collection_name=collection_name,
        persist_dir=persist_dir,
        fact_store=fact_store,
    )


def load_dual_chroma_index(
    persist_dir,
    embeddings=None,
    collection_name=None,
):
    persist_dir = Path(persist_dir)
    meta_path = persist_dir / META_FILENAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        collection_name = collection_name or meta["collection_name"]
        table_name = meta["table_collection"]
        text_name = meta["text_collection"]
    else:
        collection_name = collection_name or persist_dir.name
        table_name, text_name = dual_collection_names(collection_name)

    table_store = load_chroma_vectorstore(
        persist_dir,
        embeddings=embeddings,
        collection_name=table_name,
    )
    text_store = load_chroma_vectorstore(
        persist_dir,
        embeddings=embeddings,
        collection_name=text_name,
    )
    return DualChromaIndex(
        table_store=table_store,
        text_store=text_store,
        collection_name=collection_name,
        persist_dir=persist_dir,
        fact_store=load_fact_store(persist_dir),
    )


def has_dual_chroma_index(persist_dir, collection_name=None):
    persist_dir = Path(persist_dir)
    meta_path = persist_dir / META_FILENAME
    if meta_path.exists():
        return True

    collection_name = collection_name or persist_dir.name
    table_name, _ = dual_collection_names(collection_name)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        client.get_collection(table_name)
        return True
    except Exception:
        return False
