import re
from html.parser import HTMLParser


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._current_row = None
        self._current_cell = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._in_cell:
            text = "".join(self._current_cell).strip()
            text = re.sub(r"\s+", " ", text)
            if self._current_row is not None:
                self._current_row.append(text)
            self._in_cell = False
            self._current_cell = []
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell.append(data)


def html_table_to_markdown(html_text):
    """Convert MinerU HTML table markup into a Markdown table."""
    html_text = str(html_text or "").strip()
    if not html_text:
        return ""

    parser = _TableHTMLParser()
    parser.feed(html_text)
    rows = [row for row in parser.rows if any(cell.strip() for cell in row)]
    if not rows:
        stripped = re.sub(r"<[^>]+>", " ", html_text)
        return re.sub(r"\s+", " ", stripped).strip()

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in normalized_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_table_summary(html_text, table_caption=None, section=None):
    """Build a short retrieval-friendly summary for a financial table."""
    parser = _TableHTMLParser()
    parser.feed(str(html_text or ""))
    rows = parser.rows
    parts = []
    if table_caption:
        parts.append(str(table_caption).strip())
    if section:
        parts.append(str(section).strip())

    header_cells = []
    if rows:
        header_cells = [cell.strip() for cell in rows[0] if cell.strip()]
    if header_cells:
        parts.append("表头字段: " + "、".join(header_cells[:12]))

    metric_names = []
    for row in rows[1:]:
        if row and row[0].strip():
            metric_names.append(row[0].strip())
    if metric_names:
        parts.append("指标行: " + "、".join(metric_names[:10]))

    return "；".join(part for part in parts if part)


def enrich_table_block(block):
    """Add markdown retrieval text and summary fields to a normalized table block."""
    html_text = block.get("text_html") or block.get("text", "")
    if not str(html_text).strip().startswith("<table"):
        return block

    markdown = html_table_to_markdown(html_text)
    summary = build_table_summary(
        html_text,
        table_caption=block.get("table_caption"),
        section=block.get("section"),
    )
    enriched_parts = []
    if summary:
        enriched_parts.append(f"【表格摘要】{summary}")
    if block.get("table_caption"):
        enriched_parts.append(f"【表格标题】{block['table_caption']}")
    if markdown:
        enriched_parts.append(markdown)

    block = dict(block)
    block["text_html"] = html_text
    block["table_summary"] = summary
    block["text_markdown"] = markdown
    block["text"] = "\n\n".join(enriched_parts).strip() or html_text
    return block


_NUMERIC_QUESTION_PATTERN = re.compile(
    r"多少|几|金额|数值|比例|率|同比|环比|增长|下降|营业收入|净利润|毛利|资产|负债|现金流|每股收益|净资产|分红"
)

_QUERY_ALIASES = {
    "净利润": ["归属于上市公司股东的净利润", "净利润（元）", "净利润"],
    "归母净利润": ["归属于上市公司股东的净利润"],
    "收入": ["营业收入", "营业总收入"],
    "营收": ["营业收入", "营业总收入"],
    "营业额": ["营业收入"],
    "每股收益": ["基本每股收益", "每股收益(元/股)", "每股收益"],
    "净资产": ["归属于上市公司股东的净资产", "净资产"],
    "资产": ["资产总计", "总资产"],
    "负债": ["负债合计", "总负债"],
}


def is_numeric_finance_question(query):
    return bool(_NUMERIC_QUESTION_PATTERN.search(str(query or "")))


def rewrite_finance_query(query):
    """Expand a user query with common finance-report phrasing variants."""
    query = str(query or "").strip()
    if not query:
        return [query]

    variants = [query]
    for alias, replacements in _QUERY_ALIASES.items():
        if alias in query:
            for replacement in replacements:
                variants.append(query.replace(alias, replacement))

    deduped = []
    seen = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _tokenize(text):
    text = str(text or "").lower()
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", text))
    tokens.update(list(text))
    return {token for token in tokens if token.strip()}


def keyword_overlap_score(query, document_text):
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(document_text)
    return len(query_tokens & doc_tokens) / len(query_tokens)


def extract_relevant_table_facts(query, document_text, max_rows=5):
    """Extract row-level numeric clues from a markdown/html table for numeric QA."""
    if not is_numeric_finance_question(query):
        return ""

    query_variants = rewrite_finance_query(query)
    lines = []
    for raw_line in str(document_text).splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").replace("-", "").strip()) <= {""}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        row_text = " | ".join(cells)
        if any(term in row_text for term in query_variants) or any(
            term in row_text for term in _tokenize(query)
        ):
            lines.append(row_text)

    if not lines:
        return ""

    unique_lines = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return "；".join(unique_lines[:max_rows])
