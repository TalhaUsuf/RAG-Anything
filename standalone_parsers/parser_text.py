#!/usr/bin/env python3
"""Parse text files (.txt/.md) via ReportLab + MinerU and produce chunk JSON.

Supports: .txt, .md

For markdown files with image references (![alt](path)), images are
extracted BEFORE PDF conversion and processed separately through the
Vision LLM, matching the behavior spec in MULTIMODAL_BEHAVIOR_SPEC.md.
"""

import html as html_mod
import logging
import os
import re
import sys
from pathlib import Path

from base_parser import BaseParser, ParserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEXT_FORMATS = {".txt", ".md"}
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _escape_xml(text: str) -> str:
    return html_mod.escape(text, quote=False)


class TextParser(BaseParser):
    """Parse text files (.txt/.md) via ReportLab + MinerU."""

    def supported_formats(self) -> set[str]:
        return TEXT_FORMATS

    def preprocess(self, file_path: Path, out_dir: Path) -> tuple[Path, list[dict]]:
        content = self._read_text_file(file_path)
        extra_images: list[dict] = []

        # For markdown: extract image references before PDF conversion
        if file_path.suffix.lower() == ".md" and _MD_IMAGE_RE.search(content):
            logger.info("Extracting image references from markdown ...")
            content, extra_images = self._extract_markdown_images(content, file_path.parent)
            logger.info("Extracted %d image(s)", len(extra_images))

        logger.info("Converting %s to PDF via ReportLab ...", file_path.name)
        pdf_path = self._text_to_pdf(file_path, out_dir, content)
        return pdf_path, extra_images

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        for enc in ("utf-8", "gbk", "latin-1", "cp1252"):
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise RuntimeError(f"Could not decode {file_path}")

    @staticmethod
    def _extract_markdown_images(content: str, md_dir: Path) -> tuple[str, list[dict]]:
        images: list[dict] = []
        cleaned: list[str] = []

        for line in content.split("\n"):
            match = _MD_IMAGE_RE.search(line)
            if match:
                alt_text, img_ref = match.group(1), match.group(2)
                img_path = Path(img_ref)
                if not img_path.is_absolute():
                    img_path = (md_dir / img_path).resolve()

                if img_path.exists() and img_path.is_file():
                    images.append({
                        "type": "image",
                        "img_path": str(img_path),
                        "image_caption": [alt_text] if alt_text else [],
                        "image_footnote": [],
                        "page_idx": 0,
                    })
                    logger.info("  Image: %s (alt: %s)", img_path.name, alt_text)
                    remaining = line[:match.start()] + line[match.end():]
                    if remaining.strip():
                        cleaned.append(remaining)
                else:
                    logger.warning("  Image not found: %s — keeping as text", img_ref)
                    cleaned.append(line)
            else:
                cleaned.append(line)

        return "\n".join(cleaned), images

    @staticmethod
    def _text_to_pdf(file_path: Path, output_dir: Path, content: str) -> Path:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        except ImportError:
            logger.error("ReportLab required. Install: pip install reportlab")
            sys.exit(1)

        pdf_path = output_dir / f"{file_path.stem}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=inch, rightMargin=inch,
                                topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()

        font_name = "Helvetica"
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            wqy = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
            if os.path.exists(wqy):
                pdfmetrics.registerFont(TTFont("WenQuanYi", wqy))
                font_name = "WenQuanYi"
        except (ImportError, OSError):
            pass

        normal = ParagraphStyle("Normal2", parent=styles["Normal"],
                                fontName=font_name, fontSize=11, leading=14)
        story: list = []

        if file_path.suffix.lower() == ".md":
            for line in content.split("\n"):
                stripped = line.rstrip()
                if not stripped:
                    story.append(Spacer(1, 12))
                    continue
                if stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                    text = _escape_xml(stripped[level:].strip())
                    size = max(10, 24 - (level - 1) * 3)
                    hs = ParagraphStyle(f"H{level}", parent=styles["Normal"],
                                        fontName=font_name, fontSize=size,
                                        leading=size + 4, spaceAfter=6, spaceBefore=12)
                    try:
                        story.append(Paragraph(text, hs))
                    except Exception:
                        story.append(Paragraph(text, styles["Normal"]))
                else:
                    try:
                        story.append(Paragraph(_escape_xml(stripped), normal))
                    except Exception:
                        story.append(Paragraph(_escape_xml(stripped), styles["Normal"]))
        else:
            for line in content.split("\n"):
                stripped = line.rstrip()
                if not stripped:
                    story.append(Spacer(1, 6))
                    continue
                try:
                    story.append(Paragraph(_escape_xml(stripped), normal))
                except Exception:
                    story.append(Paragraph(_escape_xml(stripped), styles["Normal"]))

        if not story:
            story.append(Spacer(1, 6))
        doc.build(story)
        logger.info("PDF created: %s", pdf_path)
        return pdf_path


if __name__ == "__main__":
    TextParser(ParserConfig()).run_cli()
