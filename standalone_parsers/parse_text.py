#!/usr/bin/env python3
"""
Standalone text file parser for RAG-Anything.

Processes .txt and .md files through ReportLab -> PDF -> MinerU pipeline
and produces chunk JSON.

For markdown files with image references (![alt](path)), images are
extracted BEFORE PDF conversion and processed separately through the
Vision LLM, matching the behavior spec in MULTIMODAL_BEHAVIOR_SPEC.md.

No imports from the raganything package.
"""

import argparse
import html as html_mod
import json
import logging
import os
import re
import sys
from pathlib import Path

from _common import (
    build_chunks,
    enhance_caption,
    print_summary,
    read_mineru_output,
    run_mineru,
    run_mineru_remote,
    separate_content,
    unique_output_dir,
    IMAGE_CHUNK_TEMPLATE,
    compute_mdhash_id,
    split_text,
    TEXT_CHUNK_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_FORMATS = {".txt", ".md"}

# Regex to detect markdown image references: ![alt text](path/to/image.ext)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _escape_xml(text: str) -> str:
    """Escape XML entities for ReportLab Paragraph elements."""
    return html_mod.escape(text, quote=False)


# ---------------------------------------------------------------------------
# Markdown image extraction
# ---------------------------------------------------------------------------

def extract_markdown_images(content: str, md_dir: Path) -> tuple[str, list[dict]]:
    """Extract image references from markdown content.

    Returns (cleaned_content, image_items) where:
    - cleaned_content: markdown with image lines removed
    - image_items: list of dicts matching MinerU image block format
    """
    image_items: list[dict] = []
    cleaned_lines: list[str] = []

    for line in content.split("\n"):
        match = _MD_IMAGE_RE.search(line)
        if match:
            alt_text = match.group(1)
            img_ref = match.group(2)

            # Resolve path relative to the markdown file's directory
            img_path = Path(img_ref)
            if not img_path.is_absolute():
                img_path = (md_dir / img_path).resolve()

            if img_path.exists() and img_path.is_file():
                image_items.append({
                    "type": "image",
                    "img_path": str(img_path),
                    "image_caption": [alt_text] if alt_text else [],
                    "image_footnote": [],
                    "page_idx": 0,
                })
                logger.info("Extracted markdown image: %s (alt: %s)", img_path.name, alt_text)
                # Remove the image line from text content so it doesn't appear
                # as literal "![alt](path)" in the PDF
                remaining = line[:match.start()] + line[match.end():]
                if remaining.strip():
                    cleaned_lines.append(remaining)
            else:
                logger.warning("Markdown image not found: %s — keeping as text", img_ref)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines), image_items


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


def convert_text_to_pdf(file_path: str, output_dir: str, content_override: str = None) -> Path:
    """Convert a .txt or .md file to PDF using ReportLab.

    If content_override is provided, use that instead of reading the file
    (used when image references have been stripped from markdown).
    """
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

    content = content_override if content_override is not None else _read_text_file(str(file_path))
    pdf_path = output_dir / f"{file_path.stem}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
    )

    styles = getSampleStyleSheet()

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

    # Step 0: For markdown files, extract image references BEFORE PDF conversion
    md_image_items: list[dict] = []
    content_for_pdf: str | None = None

    if file_path.suffix.lower() == ".md":
        raw_content = _read_text_file(str(file_path))
        if _MD_IMAGE_RE.search(raw_content):
            logger.info("Detected image references in markdown — extracting before PDF conversion")
            content_for_pdf, md_image_items = extract_markdown_images(
                raw_content, file_path.parent
            )
            logger.info("Extracted %d image(s) from markdown", len(md_image_items))

    # Step 1: Convert to PDF (with image refs stripped if applicable)
    logger.info("Converting %s to PDF via ReportLab ...", file_path.name)
    pdf_path = convert_text_to_pdf(str(file_path), str(out_dir), content_override=content_for_pdf)

    # Step 2: Run MinerU on the PDF (extracts text, lists, equations, etc.)
    mineru_output = out_dir / "mineru"
    if args.remote:
        run_mineru_remote(str(pdf_path), str(mineru_output))
    else:
        run_mineru(str(pdf_path), str(mineru_output))

    # Step 3: Read MinerU output
    content_list = read_mineru_output(str(mineru_output), pdf_path.stem)
    logger.info("Loaded %d content items from MinerU.", len(content_list))

    # Step 4: Separate text vs multimodal from MinerU output
    full_text, mineru_multimodal = separate_content(content_list)

    # Step 5: Merge markdown-extracted images with MinerU multimodal items
    all_multimodal = mineru_multimodal + md_image_items
    if md_image_items:
        logger.info("Merged %d markdown images + %d MinerU multimodal items = %d total",
                     len(md_image_items), len(mineru_multimodal), len(all_multimodal))

    # Step 6: Build chunks (images will be sent to Vision LLM if --use-llm)
    chunks = build_chunks(full_text, all_multimodal, str(file_path), use_llm=args.use_llm)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print_summary(chunks, file_path, out_dir)


if __name__ == "__main__":
    main()
