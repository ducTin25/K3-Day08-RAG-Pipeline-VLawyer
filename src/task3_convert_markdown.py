"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

MIN_TEXT_LEN = 200  # dưới ngưỡng này coi như PDF scan (không có text layer), cần OCR

_ocr_reader = None


def _get_ocr_reader():
    """Lazy-load EasyOCR reader (chỉ tải model 1 lần, dùng lại cho mọi file)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("  (Đang tải model EasyOCR lần đầu, có thể mất một lúc...)")
        _ocr_reader = easyocr.Reader(["vi", "en"], gpu=False)
    return _ocr_reader


def _ocr_pdf(filepath: Path) -> str:
    """Render từng trang PDF thành ảnh rồi OCR — dùng cho PDF scan không có text layer."""
    import numpy as np
    import pypdfium2 as pdfium

    reader = _get_ocr_reader()
    pdf = pdfium.PdfDocument(str(filepath))
    pages_text = []
    for i, page in enumerate(pdf):
        bitmap = page.render(scale=1.5)
        image = np.array(bitmap.to_pil().convert("RGB"))
        print(f"    OCR trang {i + 1}/{len(pdf)}...")
        lines = reader.readtext(image, detail=0, paragraph=True)
        pages_text.append("\n".join(lines))
    return "\n\n".join(pages_text)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text_content = result.text_content

            if len(text_content.strip()) < MIN_TEXT_LEN and filepath.suffix.lower() == ".pdf":
                print(f"  ⚠ Text quá ngắn ({len(text_content.strip())} chars) — có thể là PDF scan, chạy OCR...")
                text_content = _ocr_pdf(filepath)

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
