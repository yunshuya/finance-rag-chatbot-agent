from pathlib import Path
import os


class MinerUParseError(RuntimeError):
    """Raised when MinerU cannot parse a PDF or produce expected output."""


def find_content_list(output_root, pdf_name):
    """Find MinerU's content_list JSON for a PDF, preferring matching stems."""
    output_root = Path(output_root)
    pdf_name = Path(pdf_name).name
    pdf_stem = Path(pdf_name).stem

    preferred_patterns = [
        f"{pdf_name}_content_list.json",
        f"{pdf_stem}_content_list.json",
        "content_list.json",
    ]
    for pattern in preferred_patterns:
        matches = sorted(output_root.rglob(pattern))
        if matches:
            return matches[0]

    matches = sorted(output_root.rglob("*content_list.json"))
    if matches:
        return matches[0]

    raise MinerUParseError(
        f"MinerU output does not contain a content_list JSON for {pdf_name}"
    )


def _configure_mineru_environment(cache_root, model_source):
    cache_root = Path(cache_root)
    os.environ.setdefault("MINERU_MODEL_SOURCE", model_source)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))


def run_mineru(
    pdf_path,
    output_root,
    backend="pipeline",
    parse_method="txt",
    lang="ch",
    formula_enable=False,
    table_enable=True,
    start_page_id=0,
    end_page_id=None,
    model_source="modelscope",
    cache_root="/private/tmp/mineru-cache",
    do_parse_func=None,
):
    """Run MinerU's sync parser and return the generated content_list JSON path."""
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    if not pdf_path.exists():
        raise MinerUParseError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise MinerUParseError(f"MinerU parser expects a PDF file: {pdf_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    _configure_mineru_environment(cache_root, model_source)

    if do_parse_func is None:
        try:
            from mineru.cli.common import do_parse as do_parse_func
        except ImportError as exc:
            raise MinerUParseError(
                "MinerU is not installed. Install it with: uv pip install -U 'mineru[all]'"
            ) from exc

    try:
        do_parse_func(
            str(output_root),
            [pdf_path.name],
            [pdf_path.read_bytes()],
            [lang],
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except Exception as exc:
        raise MinerUParseError(f"MinerU failed to parse {pdf_path.name}: {exc}") from exc

    return find_content_list(output_root, pdf_path.name)
