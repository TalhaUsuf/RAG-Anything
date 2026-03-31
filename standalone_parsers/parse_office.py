#!/usr/bin/env python3
"""
Standalone Office document parser for RAG-Anything.

Processes .doc, .docx, .ppt, .pptx, .xls, .xlsx files through
LibreOffice -> PDF -> MinerU pipeline and produces chunk JSON.

No imports from the raganything package.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}

TEXT_CHUNK_SIZE = 1200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return prefix + MD5 hex digest of content."""
    return prefix + hashlib.md5(content.encode()).hexdigest()


def unique_output_dir(base_dir, file_path):
    """Create a deterministic but unique output directory name."""
    file_path = Path(file_path).resolve()
    stem = file_path.stem
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    return Path(base_dir) / f"{stem}_{path_hash}"


# ---------------------------------------------------------------------------
# Office -> PDF conversion
# ---------------------------------------------------------------------------

def convert_office_to_pdf(doc_path: str, output_dir: str) -> Path:
    """Convert an Office document to PDF using LibreOffice."""
    doc_path = Path(doc_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if doc_path.suffix.lower() not in OFFICE_FORMATS:
        raise ValueError(f"Unsupported format: {doc_path.suffix}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        commands = ["libreoffice", "soffice"]
        converted = False

        for cmd in commands:
            try:
                args = [
                    cmd,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temp_path),
                    str(doc_path),
                ]
                kwargs = dict(capture_output=True, text=True, timeout=60)
                if platform.system() == "Windows":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                logger.info("Running: %s", " ".join(args))
                result = subprocess.run(args, **kwargs)

                if result.returncode == 0:
                    converted = True
                    break
                else:
                    logger.warning(
                        "%s failed (rc=%d): %s", cmd, result.returncode, result.stderr
                    )
            except FileNotFoundError:
                logger.debug("%s not found, trying next", cmd)
            except subprocess.TimeoutExpired:
                logger.warning("%s timed out", cmd)

        if not converted:
            raise RuntimeError(
                "LibreOffice conversion failed. Ensure libreoffice or soffice is installed."
            )

        pdfs = list(temp_path.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError("No PDF produced by LibreOffice conversion.")

        pdf_src = pdfs[0]
        if pdf_src.stat().st_size <= 100:
            raise RuntimeError("Converted PDF is too small – likely empty or corrupt.")

        pdf_dest = output_dir / pdf_src.name
        shutil.copy2(pdf_src, pdf_dest)
        logger.info("PDF saved to %s", pdf_dest)
        return pdf_dest


# ---------------------------------------------------------------------------
# MinerU processing
# ---------------------------------------------------------------------------

def _reader_thread(stream, q):
    """Read lines from a stream and put them in a queue."""
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    finally:
        q.put(None)


def run_mineru(pdf_path: str, output_dir: str) -> None:
    """Run MinerU CLI on a PDF file."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-m", "auto"]
    logger.info("Running MinerU: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    stdout_q: queue.Queue = queue.Queue()
    stderr_q: queue.Queue = queue.Queue()

    t_out = threading.Thread(target=_reader_thread, args=(proc.stdout, stdout_q), daemon=True)
    t_err = threading.Thread(target=_reader_thread, args=(proc.stderr, stderr_q), daemon=True)
    t_out.start()
    t_err.start()

    # Drain queues while process runs
    stdout_done = False
    stderr_done = False
    while not (stdout_done and stderr_done):
        if not stdout_done:
            try:
                line = stdout_q.get(timeout=0.1)
                if line is None:
                    stdout_done = True
                else:
                    logger.info("[MinerU stdout] %s", line.rstrip())
            except queue.Empty:
                pass
        if not stderr_done:
            try:
                line = stderr_q.get(timeout=0.1)
                if line is None:
                    stderr_done = True
                else:
                    logger.info("[MinerU stderr] %s", line.rstrip())
            except queue.Empty:
                pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"MinerU exited with code {proc.returncode}")
    logger.info("MinerU finished successfully.")


# ---------------------------------------------------------------------------
# Read and normalise MinerU output
# ---------------------------------------------------------------------------

def read_mineru_output(output_dir: str, file_stem: str) -> list:
    """Find and parse MinerU content_list JSON; normalise fields and paths."""
    output_dir = Path(output_dir)
    target_name = f"{file_stem}_content_list.json"

    # Search recursively
    matches = list(output_dir.rglob(target_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {target_name} under {output_dir}"
        )

    json_path = matches[0]
    logger.info("Reading MinerU output: %s", json_path)

    with open(json_path, "r", encoding="utf-8") as fh:
        content_list = json.load(fh)

    base_dir = json_path.parent

    # Normalise field aliases and fix paths
    for item in content_list:
        # img_caption -> image_caption
        if "img_caption" in item and "image_caption" not in item:
            item["image_caption"] = item.pop("img_caption")
        if "img_footnote" in item and "image_footnote" not in item:
            item["image_footnote"] = item.pop("img_footnote")

        # Fix relative image paths to absolute
        for key in ("img_path", "image_path"):
            if key in item and item[key]:
                p = Path(item[key])
                if not p.is_absolute():
                    abs_path = (base_dir / p).resolve()
                    # Security check: must stay within base_dir
                    try:
                        abs_path.relative_to(base_dir.resolve())
                    except ValueError:
                        logger.warning(
                            "Path %s escapes base directory; skipping fix", abs_path
                        )
                        continue
                    item[key] = str(abs_path)

    return content_list


# ---------------------------------------------------------------------------
# Separate content
# ---------------------------------------------------------------------------

def separate_content(content_list: list):
    """Split content list into text and multimodal items."""
    text_parts = []
    multimodal_items = []

    for item in content_list:
        if item.get("type", "text") == "text":
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            multimodal_items.append(item)

    return "\n\n".join(text_parts), multimodal_items


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def _split_text(text: str, max_len: int = TEXT_CHUNK_SIZE) -> list:
    """Split text into chunks of roughly max_len characters."""
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_len:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunks(full_text: str, multimodal_items: list, file_path: str) -> list:
    """Build chunk dicts for text and multimodal content."""
    # Compute a document-level id from all content
    signature_parts = [full_text]
    for item in multimodal_items:
        signature_parts.append(json.dumps(item, sort_keys=True, default=str))
    content_signature = "\n".join(signature_parts)
    doc_id = compute_mdhash_id(content_signature, prefix="doc-")

    chunks = []
    chunk_index = 0

    # --- Text chunks ---
    text_segments = _split_text(full_text)
    for seg in text_segments:
        chunks.append(
            {
                "content": seg,
                "tokens": len(seg.split()),
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_index,
                "file_path": os.path.basename(file_path),
                "is_multimodal": False,
            }
        )
        chunk_index += 1

    # --- Multimodal chunks ---
    image_idx = 0
    table_idx = 0
    equation_idx = 0

    for item in multimodal_items:
        item_type = item.get("type", "unknown")
        enhanced_caption = "[Requires LLM analysis]"

        if item_type == "image":
            image_idx += 1
            image_path = item.get("img_path", item.get("image_path", ""))
            captions = item.get("image_caption", "")
            footnotes = item.get("image_footnote", "")

            content = (
                f"Image Content Analysis:\n"
                f"Image Path: {image_path}\n"
                f"Captions: {captions}\n"
                f"Footnotes: {footnotes}\n"
                f"Visual Analysis: {enhanced_caption}"
            )
            entity_name = f"Image_{image_idx}"
            original_type = "image"

        elif item_type == "table":
            table_idx += 1
            table_img_path = item.get("img_path", item.get("image_path", ""))
            table_caption = item.get("table_caption", item.get("image_caption", ""))
            table_body = item.get("table_body", item.get("text", ""))
            table_footnote = item.get("table_footnote", item.get("image_footnote", ""))

            content = (
                f"Table Analysis:\n"
                f"Image Path: {table_img_path}\n"
                f"Caption: {table_caption}\n"
                f"Structure: {table_body}\n"
                f"Footnotes: {table_footnote}\n"
                f"Analysis: {enhanced_caption}"
            )
            entity_name = f"Table_{table_idx}"
            original_type = "table"

        elif item_type == "equation":
            equation_idx += 1
            equation_text = item.get("text", item.get("equation", ""))
            equation_format = item.get("equation_format", "latex")

            content = (
                f"Mathematical Equation Analysis:\n"
                f"Equation: {equation_text}\n"
                f"Format: {equation_format}\n"
                f"Mathematical Analysis: {enhanced_caption}"
            )
            entity_name = f"Equation_{equation_idx}"
            original_type = "equation"

        else:
            # Unknown multimodal type – treat generically
            content = json.dumps(item, indent=2, default=str)
            entity_name = f"{item_type.title()}_{chunk_index}"
            original_type = item_type

        chunks.append(
            {
                "content": content,
                "tokens": len(content.split()),
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_index,
                "file_path": os.path.basename(file_path),
                "is_multimodal": True,
                "modal_entity_name": entity_name,
                "original_type": original_type,
                "page_idx": item.get("page_idx", 0),
            }
        )
        chunk_index += 1

    return chunks


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
        "output_dir",
        nargs="?",
        default="./output",
        help="Output directory (default: ./output)",
    )
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    if file_path.suffix.lower() not in OFFICE_FORMATS:
        logger.error(
            "Unsupported file format: %s (expected one of %s)",
            file_path.suffix,
            ", ".join(sorted(OFFICE_FORMATS)),
        )
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Step 1: Convert to PDF
    logger.info("Converting %s to PDF ...", file_path.name)
    pdf_path = convert_office_to_pdf(str(file_path), str(out_dir))
    logger.info("PDF created: %s", pdf_path)

    # Step 2: Run MinerU
    mineru_output = out_dir / "mineru"
    logger.info("Running MinerU on %s ...", pdf_path.name)
    run_mineru(str(pdf_path), str(mineru_output))

    # Step 3: Read MinerU output
    pdf_stem = pdf_path.stem
    content_list = read_mineru_output(str(mineru_output), pdf_stem)
    logger.info("Loaded %d content items from MinerU output.", len(content_list))

    # Step 4: Separate text vs multimodal
    full_text, multimodal_items = separate_content(content_list)
    logger.info(
        "Separated into %d chars of text and %d multimodal items.",
        len(full_text),
        len(multimodal_items),
    )

    # Step 5: Build chunks
    chunks = build_chunks(full_text, multimodal_items, str(file_path))

    # Step 6: Print formatted JSON to stdout
    print(json.dumps(chunks, indent=2, ensure_ascii=False))

    # Summary to stderr so it doesn't pollute JSON stdout
    text_chunks = [c for c in chunks if not c.get("is_multimodal")]
    mm_chunks = [c for c in chunks if c.get("is_multimodal")]
    logger.info("--- Summary ---")
    logger.info("Source file:       %s", file_path)
    logger.info("Total chunks:      %d", len(chunks))
    logger.info("  Text chunks:     %d", len(text_chunks))
    logger.info("  Multimodal:      %d", len(mm_chunks))
    if mm_chunks:
        types = {}
        for c in mm_chunks:
            t = c.get("original_type", "unknown")
            types[t] = types.get(t, 0) + 1
        for t, n in sorted(types.items()):
            logger.info("    %-14s %d", t, n)
    logger.info("Output directory:  %s", out_dir)


if __name__ == "__main__":
    main()
