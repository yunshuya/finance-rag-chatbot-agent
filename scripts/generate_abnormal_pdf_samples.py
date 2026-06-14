import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "test_pdfs" / "abnormal"


def generate_samples(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_only = output_dir / "text_only_no_table_report.pdf"
    encrypted = output_dir / "encrypted_report.pdf"
    scanned = output_dir / "scanned_report.pdf"
    corrupted = output_dir / "corrupted_report.pdf"

    _write_text_only_pdf(text_only)
    _write_encrypted_pdf(text_only, encrypted)
    _write_scanned_pdf(scanned)
    _write_corrupted_pdf(corrupted)

    return {
        "text_only_no_table": text_only,
        "encrypted": encrypted,
        "scanned": scanned,
        "corrupted": corrupted,
    }


def _write_text_only_pdf(path):
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    pdf.setTitle("Text-only finance report without tables")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, height - 72, "Text-only Finance Report")
    pdf.setFont("Helvetica", 11)
    lines = [
        "This sample is a valid PDF without tables.",
        "Revenue increased steadily during the reporting period.",
        "Net profit improved because operating expenses remained controlled.",
        "No tabular structure is intentionally included in this document.",
        "Expected result: MinerU should parse text blocks and detect zero tables.",
    ]
    y = height - 110
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 22
    pdf.showPage()
    pdf.save()


def _write_encrypted_pdf(source_pdf, encrypted_pdf):
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("finance-rag-test")
    with encrypted_pdf.open("wb") as file:
        writer.write(file)


def _write_scanned_pdf(path):
    image = Image.new("RGB", (1400, 1900), "white")
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype("Arial Unicode.ttf", 46)
        font_body = ImageFont.truetype("Arial Unicode.ttf", 32)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    draw.text((110, 120), "Scanned Finance Report", fill="black", font=font_title)
    scan_lines = [
        "This page is rendered as an image, not selectable PDF text.",
        "Revenue: 100 million yuan",
        "Net profit: 20 million yuan",
        "Expected result: OCR or image parsing may be required.",
    ]
    y = 230
    for line in scan_lines:
        draw.text((110, y), line, fill="black", font=font_body)
        y += 72
    draw.rectangle((95, 95, 1305, 1680), outline="gray", width=3)

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    pdf.drawImage(
        ImageReader(image_buffer),
        36,
        36,
        width=width - 72,
        height=height - 72,
        preserveAspectRatio=True,
        anchor="c",
    )
    pdf.showPage()
    pdf.save()


def _write_corrupted_pdf(path):
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >>\n"
        b"This object is intentionally truncated and invalid.\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate abnormal PDF samples for finance RAG preprocessing tests."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where abnormal PDF samples will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = generate_samples(args.output_dir)
    for label, path in samples.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
