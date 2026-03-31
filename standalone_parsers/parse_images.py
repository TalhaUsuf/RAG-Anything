#!/usr/bin/env python3
"""
Standalone image file parser for RAG-Anything.

Processes image files (.png, .jpg, .jpeg, .bmp, .tiff, .tif, .gif, .webp)
through MinerU OCR pipeline and produces chunk JSON.

No imports from the raganything package.
"""

import argparse
import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
MINERU_NATIVE_FORMATS = {".png", ".jpeg", ".jpg"}

TEXT_CHUNK_SIZE = 1200

IMAGE_CHUNK_TEMPLATE = """Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""


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
# Image format conversion
# ---------------------------------------------------------------------------

def convert_to_png(image_path: str, output_dir: str) -> Path:
    """Convert non-native image formats to PNG for MinerU compatibility.

    MinerU natively supports .png, .jpeg, .jpg.  For other formats
    (.bmp, .tiff, .tif, .gif, .webp) we convert to PNG using PIL.

    Handles RGBA/LA/P modes by compositing onto a white background,
    and converts other modes to RGB before saving.
    """
    from PIL import Image

    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s to PNG ...", image_path.name)

    img = Image.open(image_path)

    # Handle transparency / palette modes by compositing onto white
    if img.mode in ("RGBA", "LA", "P"):
        if img.mode == "P":
            img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # use alpha channel as mask
        img = background.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    png_path = output_dir / f"{image_path.stem}.png"
    img.save(str(png_path), format="PNG", optimize=True)
    logger.info("Converted image saved to %s", png_path)
    return png_path


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


def run_mineru(input_path: str, output_dir: str, lang: str = None,
               backend: str = None, device: str = None,
               start_page: int = None, end_page: int = None) -> None:
    """Run MinerU CLI on an image file using OCR mode."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["mineru", "-p", str(input_path), "-o", str(output_dir), "-m", "ocr"]

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

    # Search recursively first
    matches = list(output_dir.rglob(target_name))

    # Fallback: check output_dir/{file_stem}/ocr/{file_stem}_content_list.json
    if not matches:
        fallback = output_dir / file_stem / "ocr" / target_name
        if fallback.exists():
            matches = [fallback]

    # Also scan output_dir/{file_stem}/ subdirectories
    if not matches:
        stem_dir = output_dir / file_stem
        if stem_dir.is_dir():
            for subdir in stem_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / target_name
                    if candidate.exists():
                        matches.append(candidate)

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

def build_text_chunks(text_content: str, doc_id: str, file_path: str,
                      chunk_size: int = TEXT_CHUNK_SIZE) -> dict:
    """Split text content into chunks and return a dict keyed by chunk ID."""
    chunks = {}
    if not text_content.strip():
        return chunks

    paragraphs = text_content.split("\n\n")
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunk_id = compute_mdhash_id(current_chunk, prefix="chunk-")
            chunks[chunk_id] = {
                "content": current_chunk.strip(),
                "tokens": len(current_chunk.split()),
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_index,
                "file_path": file_path,
            }
            chunk_index += 1
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Last chunk
    if current_chunk.strip():
        chunk_id = compute_mdhash_id(current_chunk, prefix="chunk-")
        chunks[chunk_id] = {
            "content": current_chunk.strip(),
            "tokens": len(current_chunk.split()),
            "full_doc_id": doc_id,
            "chunk_order_index": chunk_index,
            "file_path": file_path,
        }

    return chunks


def build_multimodal_chunks(multimodal_items: list, doc_id: str, file_path: str,
                            start_index: int = 0) -> dict:
    """Build chunk dicts for multimodal (image) items."""
    chunks = {}
    chunk_index = start_index

    for i, item in enumerate(multimodal_items):
        image_path = item.get("img_path", item.get("image_path", ""))
        captions = item.get("image_caption", "")
        footnotes = item.get("image_footnote", "")
        enhanced_caption = "[Requires LLM vision analysis]"

        content = IMAGE_CHUNK_TEMPLATE.format(
            image_path=image_path,
            captions=captions,
            footnotes=footnotes,
            enhanced_caption=enhanced_caption,
        )

        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        chunks[chunk_id] = {
            "content": content,
            "tokens": len(content.split()),
            "full_doc_id": doc_id,
            "chunk_order_index": chunk_index,
            "file_path": file_path,
            "is_multimodal": True,
            "modal_entity_name": f"Image_{i + 1}",
            "original_type": "image",
            "page_idx": item.get("page_idx", 0),
        }
        chunk_index += 1

    return chunks


def build_chunks(content_list: list, text_content: str, multimodal_items: list,
                 file_path: str, chunk_size: int = TEXT_CHUNK_SIZE) -> dict:
    """Build all chunks (text + multimodal) for the document."""
    # Compute doc ID from content signature
    content_signature = "\n".join([
        item.get("text", "") or item.get("img_path", "") or str(item.get("table_body", ""))
        for item in content_list
    ])
    doc_id = compute_mdhash_id(content_signature, prefix="doc-")

    # Build text chunks
    text_chunks = build_text_chunks(text_content, doc_id, file_path, chunk_size=chunk_size)

    # Determine starting chunk index for multimodal items
    next_index = len(text_chunks)

    # Build multimodal chunks
    mm_chunks = build_multimodal_chunks(
        multimodal_items, doc_id, file_path, start_index=next_index
    )

    # Merge all chunks
    all_chunks = {}
    all_chunks.update(text_chunks)
    all_chunks.update(mm_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse image files (.png/.jpg/.jpeg/.bmp/.tiff/.tif/.gif/.webp) "
        "via MinerU OCR and produce chunk JSON.",
    )
    parser.add_argument("file", help="Path to the image file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-l", "--lang",
        default=None,
        help="Language hint for MinerU OCR (e.g. 'en', 'zh')",
    )
    parser.add_argument(
        "-b", "--backend",
        default=None,
        help="MinerU backend to use",
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        help="Device for MinerU (e.g. 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=TEXT_CHUNK_SIZE,
        help=f"Target text chunk size in characters (default: {TEXT_CHUNK_SIZE})",
    )
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    suffix = file_path.suffix.lower()
    if suffix not in IMAGE_FORMATS:
        logger.error(
            "Unsupported file format: %s (expected one of %s)",
            suffix,
            ", ".join(sorted(IMAGE_FORMATS)),
        )
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Step 1: Convert to PNG if needed
    if suffix not in MINERU_NATIVE_FORMATS:
        logger.info(
            "Format %s is not natively supported by MinerU; converting to PNG ...",
            suffix,
        )
        input_path = convert_to_png(str(file_path), str(out_dir))
    else:
        input_path = file_path

    # Step 2: Run MinerU with OCR mode
    mineru_output = out_dir / "mineru"
    logger.info("Running MinerU OCR on %s ...", input_path.name)
    run_mineru(
        str(input_path),
        str(mineru_output),
        lang=args.lang,
        backend=args.backend,
        device=args.device,
    )

    # Step 3: Read MinerU output
    input_stem = input_path.stem
    content_list = read_mineru_output(str(mineru_output), input_stem)
    logger.info("Loaded %d content items from MinerU output.", len(content_list))

    # Step 4: Separate text vs multimodal
    full_text, multimodal_items = separate_content(content_list)
    logger.info(
        "Separated into %d chars of text and %d multimodal items.",
        len(full_text),
        len(multimodal_items),
    )

    # Step 5: Build chunks
    chunks = build_chunks(content_list, full_text, multimodal_items, str(file_path),
                          chunk_size=args.chunk_size)

    # Step 6: Print formatted JSON to stdout
    print(json.dumps(chunks, indent=2, ensure_ascii=False))

    # Summary to stderr so it doesn't pollute JSON stdout
    text_chunks = {k: v for k, v in chunks.items() if not v.get("is_multimodal")}
    mm_chunks = {k: v for k, v in chunks.items() if v.get("is_multimodal")}
    logger.info("--- Summary ---")
    logger.info("Source file:       %s", file_path)
    logger.info("Total chunks:      %d", len(chunks))
    logger.info("  Text chunks:     %d", len(text_chunks))
    logger.info("  Multimodal:      %d", len(mm_chunks))
    if mm_chunks:
        types = {}
        for c in mm_chunks.values():
            t = c.get("original_type", "unknown")
            types[t] = types.get(t, 0) + 1
        for t, n in sorted(types.items()):
            logger.info("    %-14s %d", t, n)
    logger.info("Output directory:  %s", out_dir)


if __name__ == "__main__":
    main()
