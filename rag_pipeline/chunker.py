from dataclasses import dataclass

try:
    from langchain.schema import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict


STRUCTURAL_BLOCK_TYPES = {
    "table",
    "equation",
    "interline_equation",
    "image",
    "chart",
    "figure",
}


def build_chunks(parsed_doc, chunk_size=1000, chunk_overlap=150):
    """Build retrieval-ready LangChain Documents from normalized MinerU JSON."""
    chunks = []
    buffer = []
    buffer_meta = None
    buffer_pages = []
    buffer_block_ids = []

    def flush_buffer():
        nonlocal buffer, buffer_meta, buffer_pages, buffer_block_ids
        if not buffer:
            return
        chunk_text = "\n".join(buffer).strip()
        if chunk_text:
            meta = dict(buffer_meta or {})
            if buffer_pages:
                meta["page_start"] = min(buffer_pages)
                meta["page_end"] = max(buffer_pages)
            if buffer_block_ids:
                meta["block_id"] = ",".join(buffer_block_ids)
            chunks.append(_make_document(parsed_doc, meta, chunk_text, len(chunks) + 1))
        buffer = []
        buffer_meta = None
        buffer_pages = []
        buffer_block_ids = []

    for block in parsed_doc.get("blocks", []):
        text = block.get("text", "").strip()
        if not text:
            continue

        block_type = block.get("type", "text")
        if block_type in STRUCTURAL_BLOCK_TYPES:
            flush_buffer()
            chunks.append(_make_document(parsed_doc, block, text, len(chunks) + 1))
            continue

        next_size = len("\n".join(buffer)) + len(text)
        if buffer_meta and block.get("section") != buffer_meta.get("section"):
            flush_buffer()
        if buffer and next_size > chunk_size:
            flush_buffer()
            if chunk_overlap > 0 and chunks:
                overlap = chunks[-1].page_content[-chunk_overlap:].strip()
                if overlap:
                    buffer.append(overlap)

        if buffer_meta is None:
            buffer_meta = block
        buffer.append(text)
        if block.get("page_no") is not None:
            buffer_pages.append(block.get("page_no"))
        if block.get("block_id"):
            buffer_block_ids.append(str(block.get("block_id")))

    flush_buffer()
    return chunks


def _make_document(parsed_doc, block, text, chunk_number):
    metadata = {
        "doc_id": parsed_doc["doc_id"],
        "source": parsed_doc["filename"],
        "page": block.get("page_no"),
        "page_start": block.get("page_start", block.get("page_no")),
        "page_end": block.get("page_end", block.get("page_no")),
        "block_type": block.get("type", "text"),
        "parser": parsed_doc.get("parser", "mineru"),
        "chunk_id": f"{parsed_doc['doc_id']}_chunk_{chunk_number}",
        "pre_chunked": True,
    }
    for field in ("block_id", "table_id", "section", "asset_path"):
        value = block.get(field)
        if value:
            metadata[field] = value

    return Document(
        page_content=text,
        metadata=metadata,
    )
