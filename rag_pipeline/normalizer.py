import json
import re
from hashlib import sha256
from collections import Counter
from pathlib import Path


SKIP_TYPES = {
    "header",
    "footer",
    "page_header",
    "page_footer",
    "page_number",
    "page_footnote",
    "aside_text",
}

TEXT_FIELDS = (
    "text",
    "content",
    "table_body",
    "latex",
    "image_caption",
    "image_footnote",
)

MEDIA_TYPES = {
    "image",
    "chart",
    "figure",
}


def normalize_mineru_content(
    content_list_path,
    filename,
    doc_id=None,
    page_offset=0,
    file_hash=None,
    source_path=None,
    parser_version=None,
    parse_status="success",
    error_message=None,
):
    """Normalize MinerU content_list output into the project's JSON shape."""
    content_list_path = Path(content_list_path)
    raw_blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
    doc_id = doc_id or make_doc_id(filename, file_hash)
    raw_page_numbers = {
        int(block.get("page_idx", 0)) + 1 + int(page_offset)
        for block in raw_blocks
        if block.get("page_idx") is not None
    }

    blocks = []
    current_section = None
    table_count = 0
    for index, block in enumerate(raw_blocks):
        block_type = str(block.get("type", "text")).strip() or "text"
        if block_type in SKIP_TYPES:
            continue

        text = _extract_text(block)
        asset_path = block.get("img_path")
        if not text and block_type in MEDIA_TYPES and asset_path:
            text = f"Image asset: {asset_path}"
        if not text:
            continue

        text_level = int(block.get("text_level", 0) or 0)
        if block_type == "text" and _looks_like_section_title(text):
            current_section = text

        normalized_block = {
            "block_id": f"{doc_id}_b{index}",
            "type": block_type,
            "text": text,
            "page_no": int(block.get("page_idx", 0)) + 1 + int(page_offset),
            "bbox": block.get("bbox"),
            "text_level": text_level,
        }
        if current_section:
            normalized_block["section"] = current_section
        if block_type == "table":
            table_count += 1
            normalized_block["table_id"] = f"{doc_id}_t{table_count}"
        if asset_path:
            normalized_block["asset_path"] = asset_path
        for source_field, target_field in (
            ("table_caption", "table_caption"),
            ("table_footnote", "table_footnote"),
            ("image_caption", "image_caption"),
            ("image_footnote", "image_footnote"),
        ):
            value = _stringify_optional(block.get(source_field))
            if value:
                normalized_block[target_field] = value

        blocks.append(normalized_block)

    block_counts = dict(sorted(Counter(block["type"] for block in blocks).items()))
    parsed_doc = {
        "doc_id": doc_id,
        "filename": filename,
        "parser": "mineru",
        "parse_status": parse_status,
        "page_offset": int(page_offset),
        "page_count": len(raw_page_numbers),
        "content_page_count": len({block["page_no"] for block in blocks}),
        "block_counts": block_counts,
        "blocks": blocks,
    }
    if file_hash:
        parsed_doc["file_hash"] = file_hash
    if source_path:
        parsed_doc["source_path"] = str(source_path)
    if parser_version:
        parsed_doc["parser_version"] = parser_version
    if error_message:
        parsed_doc["error_message"] = str(error_message)
    return parsed_doc


def write_normalized_json(parsed_doc, output_path):
    """Persist normalized parser output for reproducible preprocessing evidence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(parsed_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def compute_file_sha256(path, chunk_size=1024 * 1024):
    """Compute a stable SHA-256 hash for a source document."""
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_doc_id(filename, file_hash=None):
    """Build a stable Chroma-safe document id from filename and optional hash."""
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", stem).strip("-")
    slug = slug or "document"
    if file_hash:
        return f"{slug}-{str(file_hash)[:12]}"
    return slug


def _extract_text(block):
    for field in TEXT_FIELDS:
        value = block.get(field)
        if value is None:
            continue
        if isinstance(value, (dict, list)) and not value:
            continue
        if isinstance(value, list):
            value = " ".join(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        value = str(value).strip()
        if value:
            return value
    return ""


def _stringify_optional(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)) and not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _looks_like_section_title(text):
    text = str(text).strip()
    if not text or len(text) > 80:
        return False
    section_patterns = (
        r"^第[一二三四五六七八九十\d]+[章节]",
        r"^[一二三四五六七八九十]+、",
        r"^（[一二三四五六七八九十]+）",
        r"^\([一二三四五六七八九十\d]+\)",
        r"^\d+、",
    )
    return any(re.match(pattern, text) for pattern in section_patterns)
