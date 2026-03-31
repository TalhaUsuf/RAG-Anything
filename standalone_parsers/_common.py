"""
Shared utilities for standalone parser scripts.

Contains functions common to parse_images.py, parse_office.py,
parse_text.py, and parse_html.py so they aren't duplicated four times.
"""

import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

TEXT_CHUNK_SIZE = 1200

# Chunk templates matching raganything/prompt.py
IMAGE_CHUNK_TEMPLATE = (
    "Image Content Analysis:\n"
    "Image Path: {image_path}\n"
    "Captions: {captions}\n"
    "Footnotes: {footnotes}\n"
    "Visual Analysis: {enhanced_caption}"
)

TABLE_CHUNK_TEMPLATE = (
    "Table Analysis:\n"
    "Image Path: {table_img_path}\n"
    "Caption: {table_caption}\n"
    "Structure: {table_body}\n"
    "Footnotes: {table_footnote}\n"
    "Analysis: {enhanced_caption}"
)

EQUATION_CHUNK_TEMPLATE = (
    "Mathematical Equation Analysis:\n"
    "Equation: {equation_text}\n"
    "Format: {equation_format}\n"
    "Mathematical Analysis: {enhanced_caption}"
)

GENERIC_CHUNK_TEMPLATE = (
    "{content_type} Content Analysis:\n"
    "Content: {content}\n"
    "Analysis: {enhanced_caption}"
)


# ---------------------------------------------------------------------------
# ID / path helpers
# ---------------------------------------------------------------------------

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return *prefix* + MD5 hex digest of *content*."""
    return prefix + hashlib.md5(content.encode()).hexdigest()


def unique_output_dir(base_dir, file_path) -> Path:
    """Create a deterministic but unique output directory name."""
    file_path = Path(file_path).resolve()
    stem = file_path.stem
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    return Path(base_dir) / f"{stem}_{path_hash}"


# ---------------------------------------------------------------------------
# MinerU execution
# ---------------------------------------------------------------------------

def _reader_thread(stream, q):
    """Read lines from *stream* and put them in *q*; signal end with None."""
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    finally:
        q.put(None)


def run_mineru(
    input_path: str,
    output_dir: str,
    *,
    mode: str = "auto",
    lang: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> None:
    """Run MinerU CLI on *input_path* and write output to *output_dir*."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["mineru", "-p", str(input_path), "-o", str(output_dir), "-m", mode]
    if lang:
        cmd.extend(["-l", lang])
    if backend:
        cmd.extend(["-b", backend])
    if device:
        cmd.extend(["-d", device])
    if start_page is not None:
        cmd.extend(["-s", str(start_page)])
    if end_page is not None:
        cmd.extend(["-e", str(end_page)])

    logger.info("Running MinerU: %s", " ".join(cmd))

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as proc:
        stdout_q: queue.Queue = queue.Queue()
        stderr_q: queue.Queue = queue.Queue()

        t_out = threading.Thread(
            target=_reader_thread, args=(proc.stdout, stdout_q), daemon=True
        )
        t_err = threading.Thread(
            target=_reader_thread, args=(proc.stderr, stderr_q), daemon=True
        )
        t_out.start()
        t_err.start()

        stdout_done = stderr_done = False
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
    """Find and parse MinerU ``content_list`` JSON; normalise fields and paths."""
    output_dir = Path(output_dir)
    target_name = f"{file_stem}_content_list.json"

    matches = list(output_dir.rglob(target_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {target_name} under {output_dir}"
        )

    json_path = matches[0]
    logger.info("Reading MinerU output: %s", json_path)

    with open(json_path, "r", encoding="utf-8") as fh:
        content_list = json.load(fh)

    base_dir = json_path.parent.resolve()

    for item in content_list:
        # Normalise field aliases (MinerU 1.x → 2.0)
        if "img_caption" in item and "image_caption" not in item:
            item["image_caption"] = item.pop("img_caption")
        if "img_footnote" in item and "image_footnote" not in item:
            item["image_footnote"] = item.pop("img_footnote")

        # Fix relative image paths to absolute; skip items that escape base_dir
        for key in ("img_path", "image_path"):
            if key in item and item[key]:
                p = Path(item[key])
                if not p.is_absolute():
                    abs_path = (base_dir / p).resolve()
                    try:
                        abs_path.relative_to(base_dir)
                    except ValueError:
                        logger.warning(
                            "Path traversal blocked for %s – clearing field", item[key]
                        )
                        item[key] = ""
                        continue
                    item[key] = str(abs_path)

    return content_list


# ---------------------------------------------------------------------------
# Content separation
# ---------------------------------------------------------------------------

def separate_content(content_list: list) -> Tuple[str, list]:
    """Split *content_list* into merged text string and multimodal items."""
    text_parts: list[str] = []
    multimodal_items: list[dict] = []

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

def split_text(text: str, max_len: int = TEXT_CHUNK_SIZE) -> List[str]:
    """Split *text* into chunks of roughly *max_len* characters."""
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
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


def _format_multimodal_chunk(item: dict) -> Tuple[str, str, str]:
    """Return (content, entity_name, original_type) for a multimodal item.

    Uses the same templates as raganything/prompt.py.
    """
    item_type = item.get("type", "unknown")
    enhanced_caption = "[Requires LLM analysis]"

    if item_type == "image":
        content = IMAGE_CHUNK_TEMPLATE.format(
            image_path=item.get("img_path", item.get("image_path", "")),
            captions=item.get("image_caption", ""),
            footnotes=item.get("image_footnote", ""),
            enhanced_caption=enhanced_caption,
        )
        return content, "image", "image"

    if item_type == "table":
        content = TABLE_CHUNK_TEMPLATE.format(
            table_img_path=item.get("img_path", item.get("image_path", "")),
            table_caption=item.get("table_caption", item.get("image_caption", "")),
            table_body=item.get("table_body", item.get("text", "")),
            table_footnote=item.get("table_footnote", item.get("image_footnote", "")),
            enhanced_caption=enhanced_caption,
        )
        return content, "table", "table"

    if item_type == "equation":
        content = EQUATION_CHUNK_TEMPLATE.format(
            equation_text=item.get("text", item.get("equation", "")),
            equation_format=item.get("equation_format", "latex"),
            enhanced_caption=enhanced_caption,
        )
        return content, "equation", "equation"

    # Generic / unknown type
    content = GENERIC_CHUNK_TEMPLATE.format(
        content_type=item_type.title(),
        content=json.dumps(item, indent=2, default=str),
        enhanced_caption=enhanced_caption,
    )
    return content, item_type, item_type


def build_chunks(
    full_text: str,
    multimodal_items: list,
    file_path: str,
    *,
    chunk_size: int = TEXT_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """Build chunk dicts for text and multimodal content."""
    # Document-level ID
    sig_parts = [full_text]
    for item in multimodal_items:
        sig_parts.append(json.dumps(item, sort_keys=True, default=str))
    doc_id = compute_mdhash_id("\n".join(sig_parts), prefix="doc-")

    chunks: list[dict] = []
    chunk_index = 0
    basename = os.path.basename(file_path)

    # --- text chunks ---
    for seg in split_text(full_text, max_len=chunk_size):
        chunks.append(
            {
                "content": seg,
                "tokens": len(seg.split()),
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_index,
                "file_path": basename,
                "is_multimodal": False,
            }
        )
        chunk_index += 1

    # --- multimodal chunks ---
    type_counters: dict[str, int] = {}
    for item in multimodal_items:
        content, entity_base, original_type = _format_multimodal_chunk(item)
        type_counters[entity_base] = type_counters.get(entity_base, 0) + 1
        entity_name = f"{entity_base.title()}_{type_counters[entity_base]}"

        chunks.append(
            {
                "content": content,
                "tokens": len(content.split()),
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_index,
                "file_path": basename,
                "is_multimodal": True,
                "modal_entity_name": entity_name,
                "original_type": original_type,
                "page_idx": item.get("page_idx", 0),
            }
        )
        chunk_index += 1

    return chunks


def print_summary(chunks: list, file_path, out_dir) -> None:
    """Log a human-readable summary of the generated chunks."""
    text_chunks = [c for c in chunks if not c.get("is_multimodal")]
    mm_chunks = [c for c in chunks if c.get("is_multimodal")]
    logger.info("--- Summary ---")
    logger.info("Source file:       %s", file_path)
    logger.info("Total chunks:      %d", len(chunks))
    logger.info("  Text chunks:     %d", len(text_chunks))
    logger.info("  Multimodal:      %d", len(mm_chunks))
    if mm_chunks:
        types: dict[str, int] = {}
        for c in mm_chunks:
            t = c.get("original_type", "unknown")
            types[t] = types.get(t, 0) + 1
        for t, n in sorted(types.items()):
            logger.info("    %-14s %d", t, n)
    logger.info("Output directory:  %s", out_dir)


# ---------------------------------------------------------------------------
# LibreOffice PDF conversion (used by Office + HTML parsers)
# ---------------------------------------------------------------------------

def convert_to_pdf_via_libreoffice(source_path: str, output_dir: str) -> Path:
    """Convert a document to PDF using LibreOffice (headless)."""
    import platform as _platform
    import shutil
    import tempfile

    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        commands_to_try = ["libreoffice", "soffice"]
        success = False

        for cmd in commands_to_try:
            try:
                convert_cmd = [
                    cmd, "--headless", "--convert-to", "pdf",
                    "--outdir", str(tmp_path), str(source_path),
                ]
                kwargs: dict = {
                    "capture_output": True, "text": True,
                    "timeout": 60, "encoding": "utf-8", "errors": "ignore",
                }
                if _platform.system() == "Windows":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                result = subprocess.run(convert_cmd, **kwargs)
                if result.returncode == 0:
                    success = True
                    logger.info("Converted %s to PDF via %s", source_path.name, cmd)
                    break
                logger.warning("%s failed: %s", cmd, result.stderr)
            except FileNotFoundError:
                logger.warning("Command %s not found", cmd)
            except subprocess.TimeoutExpired:
                logger.warning("Command %s timed out", cmd)

        if not success:
            raise RuntimeError(
                f"LibreOffice conversion failed for {source_path.name}. "
                "Ensure LibreOffice is installed."
            )

        pdfs = list(tmp_path.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError(
                f"No PDF generated for {source_path.name}"
            )
        pdf = pdfs[0]
        if pdf.stat().st_size < 100:
            raise RuntimeError("Generated PDF appears empty or corrupt.")

        final = output_dir / f"{source_path.stem}.pdf"
        shutil.copy2(pdf, final)

    return final
