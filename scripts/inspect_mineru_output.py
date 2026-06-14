import argparse
import json
from collections import Counter
from pathlib import Path


TABLE_TYPES = {"table"}
IMAGE_TYPES = {"image", "figure", "chart"}
TEXT_TYPES = {"text"}


def inspect_content_list(content_list_path):
    content_list_path = Path(content_list_path)
    blocks = json.loads(content_list_path.read_text(encoding="utf-8"))

    type_counts = Counter(str(block.get("type", "text")) for block in blocks)
    pages = sorted(
        {
            int(block["page_idx"]) + 1
            for block in blocks
            if block.get("page_idx") is not None
        }
    )

    table_blocks = [
        block for block in blocks if str(block.get("type", "text")) in TABLE_TYPES
    ]
    image_blocks = [
        block for block in blocks if str(block.get("type", "text")) in IMAGE_TYPES
    ]

    return {
        "file": str(content_list_path),
        "total_blocks": len(blocks),
        "type_counts": dict(sorted(type_counts.items())),
        "page_count": len(pages),
        "pages": pages,
        "table_count": len(table_blocks),
        "tables_with_body": sum(1 for block in table_blocks if block.get("table_body")),
        "tables_with_asset": sum(1 for block in table_blocks if block.get("img_path")),
        "image_count": len(image_blocks),
        "images_with_asset": sum(1 for block in image_blocks if block.get("img_path")),
        "images_with_caption": sum(
            1 for block in image_blocks if _has_text(block.get("image_caption"))
        ),
        "missing_page_blocks": sum(1 for block in blocks if block.get("page_idx") is None),
        "empty_text_blocks": sum(
            1
            for block in blocks
            if str(block.get("type", "text")) in TEXT_TYPES
            and not _has_text(block.get("text"))
        ),
        "table_samples": [
            _sample(block.get("table_body") or block.get("text") or "")
            for block in table_blocks[:3]
        ],
        "image_samples": [
            {
                "page": int(block.get("page_idx", 0)) + 1,
                "asset": block.get("img_path"),
                "caption": _sample(block.get("image_caption")),
            }
            for block in image_blocks[:3]
        ],
    }


def format_report(report):
    lines = [
        f"File: {report['file']}",
        f"Total blocks: {report['total_blocks']}",
        f"Pages covered: {report['page_count']} ({_format_pages(report['pages'])})",
        f"Type counts: {json.dumps(report['type_counts'], ensure_ascii=False)}",
        (
            "Tables: "
            f"{report['table_count']} "
            f"(with table_body: {report['tables_with_body']}, "
            f"with asset: {report['tables_with_asset']})"
        ),
        (
            "Images/Figures/Charts: "
            f"{report['image_count']} "
            f"(with asset: {report['images_with_asset']}, "
            f"with caption: {report['images_with_caption']})"
        ),
        f"Blocks missing page_idx: {report['missing_page_blocks']}",
        f"Empty text blocks: {report['empty_text_blocks']}",
    ]

    if report["table_samples"]:
        lines.append("Table samples:")
        lines.extend(f"- {sample}" for sample in report["table_samples"])

    if report["image_samples"]:
        lines.append("Image samples:")
        lines.extend(
            f"- page {sample['page']}: asset={sample['asset']}, caption={sample['caption']}"
            for sample in report["image_samples"]
        )

    return "\n".join(lines)


def _has_text(value):
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return bool(value)
    return bool(str(value).strip())


def _sample(value, limit=180):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_pages(pages):
    if not pages:
        return "none"
    if len(pages) <= 12:
        return ", ".join(str(page) for page in pages)
    start = ", ".join(str(page) for page in pages[:6])
    end = ", ".join(str(page) for page in pages[-6:])
    return f"{start}, ..., {end}"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect MinerU content_list.json for finance report preprocessing."
    )
    parser.add_argument("content_list", help="Path to MinerU content_list.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    parser.add_argument(
        "--require-table",
        action="store_true",
        help="Exit with an error if no table block is detected.",
    )
    parser.add_argument(
        "--require-image",
        action="store_true",
        help="Exit with an error if no image/figure/chart block is detected.",
    )
    args = parser.parse_args()

    report = inspect_content_list(args.content_list)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    failures = []
    if args.require_table and report["table_count"] == 0:
        failures.append("required table block was not detected")
    if args.require_image and report["image_count"] == 0:
        failures.append("required image/figure/chart block was not detected")

    if failures:
        parser.exit(2, "ERROR: " + "; ".join(failures) + "\n")


if __name__ == "__main__":
    main()
