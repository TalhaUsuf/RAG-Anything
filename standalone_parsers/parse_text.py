#!/usr/bin/env python3
"""
Standalone text file parser for RAG-Anything.

Processes .txt and .md files through ReportLab -> PDF -> MinerU pipeline
and produces chunk JSON.

No imports from the raganything package.
"""

import argparse
import html as html_mod
import json
import logging
import os
import sys
from pathlib import Path

from _common import (
    build_chunks,
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

TEXT_FORMATS = {".txt", ".md"}


def _escape_xml(text: str) -> str:
    """Escape XML entities for ReportLab Paragraph elements."""
    return html_mod.escape(text, quote=False)


# ---------------------------------------------------------------------------
# Text -> PDF via ReportLab
# ---------------------------------------------------------------------------

def _read_text_file(file_path: str) -> str:
    """Read a text file trying multiple encodings."""
    for enc in ("utf-8", "gbk", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Could not decode {file_path}")


def convert_text_to_pdf(file_path: str, output_dir: str) -> Path:
    """Convert a .txt or .md file to PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        logger.error(
            "ReportLab is required for text-to-PDF conversion. "
            "Install it with: pip install reportlab"
        )
        sys.exit(1)

    file_path = Path(file_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content = _read_text_file(str(file_path))
    pdf_path = output_dir / f"{file_path.stem}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
    )

    styles = getSampleStyleSheet()

    # Try to register WenQuanYi font for Chinese support
    font_name = "Helvetica"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        wqy_path = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
        if os.path.exists(wqy_path):
            pdfmetrics.registerFont(TTFont("WenQuanYi", wqy_path))
            font_name = "WenQuanYi"
    except (ImportError, OSError) as exc:
        logger.debug("Could not register WenQuanYi font: %s", exc)

    normal_style = ParagraphStyle(
        "NormalCustom", parent=styles["Normal"],
        fontName=font_name, fontSize=11, leading=14,
    )

    story = []
    lines = content.split("\n")

    if file_path.suffix.lower() == ".md":
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                story.append(Spacer(1, 12))
                continue

            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                header_text = _escape_xml(stripped[level:].strip())
                font_size = max(10, 24 - (level - 1) * 3)
                heading_style = ParagraphStyle(
                    f"Heading{level}", parent=styles["Normal"],
                    fontName=font_name, fontSize=font_size,
                    leading=font_size + 4, spaceAfter=6, spaceBefore=12,
                )
                try:
                    story.append(Paragraph(header_text, heading_style))
                except Exception as exc:
                    logger.debug("Heading style failed, using fallback: %s", exc)
                    story.append(Paragraph(header_text, styles["Normal"]))
            else:
                try:
                    story.append(Paragraph(_escape_xml(stripped), normal_style))
                except Exception as exc:
                    logger.debug("Paragraph style failed, using fallback: %s", exc)
                    story.append(Paragraph(_escape_xml(stripped), styles["Normal"]))
    else:
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                story.append(Spacer(1, 6))
                continue
            try:
                story.append(Paragraph(_escape_xml(stripped), normal_style))
            except Exception:
                story.append(Paragraph(_escape_xml(stripped), styles["Normal"]))

    if not story:
        story.append(Spacer(1, 6))

    doc.build(story)
    logger.info("PDF created: %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse text files (.txt/.md) via ReportLab + MinerU and produce chunk JSON.",
    )
    parser.add_argument("file", help="Path to the text file (.txt or .md)")
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

    if file_path.suffix.lower() not in TEXT_FORMATS:
        logger.error("Unsupported format: %s (expected one of %s)",
                      file_path.suffix, ", ".join(sorted(TEXT_FORMATS)))
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to PDF
    logger.info("Converting %s to PDF via ReportLab ...", file_path.name)
    pdf_path = convert_text_to_pdf(str(file_path), str(out_dir))

    # Step 2: Run MinerU
    mineru_output = out_dir / "mineru"
    if args.remote:
        run_mineru_remote(str(pdf_path), str(mineru_output))
    else:
        run_mineru(str(pdf_path), str(mineru_output))

    # Step 3: Read MinerU output
    content_list = read_mineru_output(str(mineru_output), pdf_path.stem)
    logger.info("Loaded %d content items.", len(content_list))

    # Step 4: Separate text vs multimodal
    full_text, multimodal_items = separate_content(content_list)

    # Step 5: Build and output chunks
    chunks = build_chunks(full_text, multimodal_items, str(file_path), use_llm=args.use_llm)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print_summary(chunks, file_path, out_dir)


if __name__ == "__main__":
    main()
