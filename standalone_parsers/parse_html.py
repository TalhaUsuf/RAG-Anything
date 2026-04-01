#!/usr/bin/env python3
"""
Standalone HTML document parser for RAG-Anything.

Processes .html, .htm, .xhtml files through
LibreOffice -> PDF -> MinerU pipeline and produces chunk JSON.

No imports from the raganything package.

NOTE ON PARSING APPROACHES:
- In the original RAG-Anything codebase, DoclingParser has native HTML support
  via its parse_html() method, which can parse HTML structure directly.
- However, at the processor level (processor.py), HTML files are grouped with
  Office files and sent through the LibreOffice conversion path.
- This script uses the LibreOffice -> PDF -> MinerU path, which is lossy
  (HTML structure, links, and semantic markup are lost in the PDF conversion)
  but universally available wherever LibreOffice is installed.
- For better HTML parsing that preserves structure and links, consider using
  Docling directly.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from _common import (
    build_chunks,
    convert_to_pdf_via_libreoffice,
    print_summary,
    read_mineru_output,
    run_mineru,
    run_mineru_remote,
    separate_content,
    unique_output_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HTML_FORMATS = {".html", ".htm", ".xhtml"}


def _html_to_pdf_fallback(html_path: str, output_dir: str) -> Path:
    """Fallback HTML→PDF conversion using ReportLab when LibreOffice is unavailable.

    Strips HTML tags and renders plain text content as a PDF.
    """
    import html as html_mod
    import re

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        logger.error("ReportLab is required as fallback. Install with: pip install reportlab")
        sys.exit(1)

    html_path = Path(html_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(html_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip HTML tags, decode entities
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_mod.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    pdf_path = output_dir / f"{html_path.stem}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []
    for para in text.split(". "):
        para = para.strip()
        if para:
            safe = html_mod.escape(para, quote=False)
            story.append(Paragraph(safe, styles["Normal"]))
            story.append(Spacer(1, 6))
    if not story:
        story.append(Spacer(1, 6))
    doc.build(story)
    logger.info("HTML fallback PDF created: %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse HTML documents (.html/.htm/.xhtml) "
        "via LibreOffice + MinerU and produce chunk JSON.",
    )
    parser.add_argument("file", help="Path to the HTML document")
    parser.add_argument(
        "output_dir", nargs="?", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument("--remote", action="store_true", help="Use remote MinerU API instead of local CLI")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for enhanced captions")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    if file_path.suffix.lower() not in HTML_FORMATS:
        logger.error("Unsupported format: %s (expected one of %s)",
                      file_path.suffix, ", ".join(sorted(HTML_FORMATS)))
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    mineru_output = out_dir / "mineru"

    # Convert HTML to PDF first (MinerU API doesn't accept HTML directly)
    logger.info("Converting %s to PDF ...", file_path.name)
    try:
        pdf_path = convert_to_pdf_via_libreoffice(str(file_path), str(out_dir))
    except RuntimeError:
        # Fallback: use weasyprint or simple text extraction
        logger.warning("LibreOffice failed; falling back to direct text extraction from HTML")
        pdf_path = _html_to_pdf_fallback(str(file_path), str(out_dir))

    if args.remote:
        run_mineru_remote(str(pdf_path), str(mineru_output))
    else:
        run_mineru(str(pdf_path), str(mineru_output))
    stem_for_lookup = pdf_path.stem

    # Read MinerU output
    content_list = read_mineru_output(str(mineru_output), stem_for_lookup)
    logger.info("Loaded %d content items.", len(content_list))

    # Step 4: Separate text vs multimodal
    full_text, multimodal_items = separate_content(content_list)

    # Step 5: Build and output chunks
    chunks = build_chunks(full_text, multimodal_items, str(file_path), use_llm=args.use_llm)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print_summary(chunks, file_path, out_dir)


if __name__ == "__main__":
    main()
