#!/usr/bin/env python3
"""
Standalone Office document parser for RAG-Anything.

Processes .doc, .docx, .ppt, .pptx, .xls, .xlsx files through
LibreOffice -> PDF -> MinerU pipeline and produces chunk JSON.

No imports from the raganything package.
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

OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse Office documents (.doc/.docx/.ppt/.pptx/.xls/.xlsx) "
        "via LibreOffice + MinerU and produce chunk JSON.",
    )
    parser.add_argument("file", help="Path to the Office document")
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

    if file_path.suffix.lower() not in OFFICE_FORMATS:
        logger.error("Unsupported format: %s (expected one of %s)",
                      file_path.suffix, ", ".join(sorted(OFFICE_FORMATS)))
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to PDF
    logger.info("Converting %s to PDF ...", file_path.name)
    pdf_path = convert_to_pdf_via_libreoffice(str(file_path), str(out_dir))

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
