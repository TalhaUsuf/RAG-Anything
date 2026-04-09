#!/usr/bin/env python3
"""Parse HTML documents via LibreOffice/ReportLab + MinerU and produce chunk JSON.

Supports: .html, .htm, .xhtml

NOTE: LibreOffice conversion is attempted first. If it fails, a ReportLab
fallback strips HTML tags and renders plain text as PDF. This is lossy —
for structure-preserving HTML parsing, consider using Docling directly.
"""

import html as html_mod
import logging
import re
import sys
from pathlib import Path

from _common import convert_to_pdf_via_libreoffice
from base_parser import BaseParser, ParserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HTML_FORMATS = {".html", ".htm", ".xhtml"}


class HtmlParser(BaseParser):
    """Parse HTML documents (.html/.htm/.xhtml) via LibreOffice + MinerU."""

    def supported_formats(self) -> set[str]:
        return HTML_FORMATS

    def preprocess(self, file_path: Path, out_dir: Path) -> tuple[Path, list[dict]]:
        logger.info("Converting %s to PDF ...", file_path.name)
        try:
            pdf_path = convert_to_pdf_via_libreoffice(str(file_path), str(out_dir))
        except RuntimeError:
            logger.warning("LibreOffice failed; falling back to ReportLab text extraction")
            pdf_path = self._html_to_pdf_fallback(file_path, out_dir)
        return pdf_path, []

    @staticmethod
    def _html_to_pdf_fallback(html_path: Path, output_dir: Path) -> Path:
        """Strip HTML tags, render plain text as PDF via ReportLab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        except ImportError:
            logger.error("ReportLab required as fallback. Install: pip install reportlab")
            sys.exit(1)

        with open(html_path, "r", encoding="utf-8") as f:
            raw = f.read()

        text = re.sub(r"<[^>]+>", " ", raw)
        text = html_mod.unescape(text)
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
                story.append(Paragraph(html_mod.escape(para, quote=False), styles["Normal"]))
                story.append(Spacer(1, 6))
        if not story:
            story.append(Spacer(1, 6))
        doc.build(story)
        logger.info("HTML fallback PDF created: %s", pdf_path)
        return pdf_path


if __name__ == "__main__":
    HtmlParser(ParserConfig()).run_cli()
