#!/usr/bin/env python3
"""Parse Office documents via LibreOffice + MinerU and produce chunk JSON.

Supports: .doc, .docx, .ppt, .pptx, .xls, .xlsx
"""

import logging
from pathlib import Path

from _common import convert_to_pdf_via_libreoffice
from base_parser import BaseParser, ParserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}


class OfficeParser(BaseParser):
    """Parse Office documents (.doc/.docx/.ppt/.pptx/.xls/.xlsx) via LibreOffice + MinerU."""

    def supported_formats(self) -> set[str]:
        return OFFICE_FORMATS

    def preprocess(self, file_path: Path, out_dir: Path) -> tuple[Path, list[dict]]:
        logger.info("Converting %s to PDF via LibreOffice ...", file_path.name)
        pdf_path = convert_to_pdf_via_libreoffice(str(file_path), str(out_dir))
        return pdf_path, []


if __name__ == "__main__":
    OfficeParser(ParserConfig()).run_cli()
