"""
Shared utilities for standalone parser scripts.

Contains functions common to parse_images.py, parse_office.py,
parse_text.py, and parse_html.py so they aren't duplicated four times.
"""

import hashlib
import io
import json
import logging
import mimetypes
import os
import queue
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

TEXT_CHUNK_SIZE = 1200

# ── Remote MinerU API defaults ──────────────────────────────────────────────
MINERU_API_URL = os.environ.get("MINERU_API_URL", "http://69.48.159.8:40050")
MINERU_ENDPOINT = "/file_parse"

# ── LLM defaults (OpenAI-compatible) ───────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://69.48.159.10:30000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-70b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# ── Vision LLM defaults (OpenAI-compatible multimodal) ────────────────────
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://69.48.159.8:23333/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "OpenGVLab/InternVL3-38B")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")

# ── Embedding defaults ─────────────────────────────────────────────────────
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://69.48.159.8:30007/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Nexus_Embedding_Model_seq_8192_embd_1024")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")

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
# Remote MinerU API
# ---------------------------------------------------------------------------

def run_mineru_remote(
    input_path: str,
    output_dir: str,
    *,
    parse_method: str = "auto",
    lang: str = "en",
    backend: str = "hybrid-auto-engine",
    api_url: str | None = None,
) -> None:
    """POST a file to the remote MinerU HTTP API and extract the ZIP response.

    The API returns a ZIP containing ``*_content_list.json``, markdown, and
    images — the same layout that the local ``mineru`` CLI produces.
    """
    api_url = api_url or MINERU_API_URL
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    filename = input_path.name
    content_type = mimetypes.guess_type(str(input_path))[0] or "application/octet-stream"

    # Build multipart/form-data body
    boundary = f"----MineruBoundary{hashlib.md5(str(time.time()).encode()).hexdigest()[:16]}"
    body_parts: list[bytes] = []

    form_fields = {
        "return_middle_json": "true",
        "return_model_output": "true",
        "return_md": "true",
        "return_images": "true",
        "return_content_list": "true",
        "response_format_zip": "true",
        "table_enable": "true",
        "formula_enable": "true",
        "parse_method": parse_method,
        "backend": backend,
        "lang_list": lang,
        "start_page_id": "0",
        "end_page_id": "99999",
        "output_dir": "./output",
        "server_url": "string",
    }

    for key, value in form_fields.items():
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{key}"\r\n'
            f"Content-Type: text/plain\r\n\r\n"
            f"{value}\r\n".encode()
        )

    # File field
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
    )
    with open(input_path, "rb") as f:
        body_parts.append(f.read())
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_parts)

    url = f"{api_url}{MINERU_ENDPOINT}"
    logger.info("POSTing %s (%d bytes) to %s", filename, len(body), url)

    req = Request(
        url,
        data=body,
        headers={
            "accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=600) as resp:
            status = resp.status
            resp_data = resp.read()
            resp_content_type = resp.headers.get("Content-Type", "")
    except Exception as exc:
        raise RuntimeError(f"MinerU API request failed: {exc}") from exc

    logger.info("MinerU API responded: HTTP %d, %d bytes, Content-Type: %s",
                status, len(resp_data), resp_content_type)

    if status != 200:
        raw_out = output_dir / "error_response.bin"
        raw_out.write_bytes(resp_data)
        raise RuntimeError(
            f"MinerU API returned HTTP {status}. Response saved to {raw_out}"
        )

    # Handle ZIP response
    if zipfile.is_zipfile(io.BytesIO(resp_data)):
        try:
            with zipfile.ZipFile(io.BytesIO(resp_data)) as zf:
                zf.extractall(output_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError(f"Failed to extract MinerU ZIP response: {exc}") from exc
        logger.info("Extracted ZIP to %s (%d files)", output_dir,
                     len(list(output_dir.rglob("*"))))
    else:
        # Non-ZIP response — try to parse as JSON and save content_list
        try:
            result = json.loads(resp_data)
        except json.JSONDecodeError:
            raw_out = output_dir / "raw_response.bin"
            raw_out.write_bytes(resp_data)
            raise RuntimeError(
                f"MinerU API returned non-ZIP, non-JSON response ({len(resp_data)} bytes). "
                f"Saved to {raw_out}"
            )

        # Normalize to content_list
        content = (
            result if isinstance(result, list)
            else result.get("content_list") if isinstance(result, dict)
            else None
        )
        stem = input_path.stem
        if content is not None:
            json_out = output_dir / f"{stem}_content_list.json"
        else:
            json_out = output_dir / "raw_response.json"
            content = result
            logger.warning("Unexpected JSON structure; saved to %s", json_out)

        json_out.write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info("Remote MinerU processing complete.")


# ---------------------------------------------------------------------------
# LLM integration (OpenAI-compatible)
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint using only stdlib."""
    base_url = (base_url or LLM_BASE_URL).rstrip("/")
    model = model or LLM_MODEL
    api_key = api_key if api_key is not None else LLM_API_KEY
    url = f"{base_url}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(url, data=payload, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        if not content:
            logger.warning("LLM returned empty content")
            return "[LLM returned empty response]"
        return content
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("LLM response has unexpected schema: %s", exc)
        return "[LLM analysis unavailable]"
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return "[LLM analysis unavailable]"


def call_vlm(
    prompt: str,
    image_base64: str,
    *,
    system_prompt: str = "You are an expert image analyst. Provide detailed, accurate descriptions.",
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    image_mime: str = "image/png",
) -> str:
    """Call a Vision LLM with base64 image using OpenAI-compatible multimodal format."""
    base_url = (base_url or VLM_BASE_URL).rstrip("/")
    model = model or VLM_MODEL
    api_key = api_key if api_key is not None else VLM_API_KEY
    url = f"{base_url}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{image_mime};base64,{image_base64}",
                }},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(url, data=payload, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        if not content:
            return "[Vision LLM returned empty response]"
        return content
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Vision LLM response has unexpected schema: %s", exc)
        return "[Vision LLM analysis unavailable]"
    except Exception as exc:
        logger.warning("Vision LLM call failed: %s", exc)
        return "[Vision LLM analysis unavailable]"


def _encode_image_file(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, mime_type)."""
    import base64
    path = Path(image_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return data, mime


def enhance_caption(item: dict, item_type: str, context: str = "") -> str:
    """Use the appropriate LLM to generate an enhanced caption.

    Images are sent to the Vision LLM with base64-encoded image data.
    All other types use the text LLM.
    """
    if item_type == "image":
        img_path = item.get("img_path", item.get("image_path", ""))
        captions = item.get("image_caption", "N/A")
        footnotes = item.get("image_footnote", "N/A")

        prompt = (
            "Please analyze this image in detail and provide a JSON response:\n"
            '{"detailed_description": "comprehensive visual description",'
            ' "entity_info": {"entity_name": "descriptive name",'
            ' "entity_type": "image", "summary": "max 100 words"}}\n\n'
            f"Image Path: {img_path}\n"
            f"Captions: {captions}\nFootnotes: {footnotes}\n"
        )
        if context:
            prompt += f"\nContext from surrounding content:\n{context}\n"

        # Try to send actual image to Vision LLM
        if img_path and Path(img_path).exists():
            try:
                b64, mime = _encode_image_file(img_path)
                return call_vlm(prompt, b64, image_mime=mime)
            except Exception as exc:
                logger.warning("Vision LLM failed for %s: %s — falling back to text LLM",
                               img_path, exc)

        # Fallback to text LLM if image not available
        return call_llm(prompt, system_prompt="You are an expert image analyst.")

    elif item_type == "table":
        prompt = (
            "Analyze this table content. Provide a JSON response:\n"
            '{"detailed_description": "table structure, headers, data patterns, insights",'
            ' "entity_info": {"entity_name": "name", "entity_type": "table",'
            ' "summary": "max 100 words"}}\n\n'
            f"Caption: {item.get('table_caption', 'N/A')}\n"
            f"Body: {item.get('table_body', 'N/A')}\n"
            f"Footnotes: {item.get('table_footnote', 'N/A')}\n"
        )
        if context:
            prompt += f"\nContext:\n{context}\n"
        return call_llm(prompt, system_prompt="You are an expert data analyst.")

    elif item_type == "equation":
        prompt = (
            "Analyze this equation. Provide a JSON response:\n"
            '{"detailed_description": "meaning, variables, applications",'
            ' "entity_info": {"entity_name": "name", "entity_type": "equation",'
            ' "summary": "max 100 words"}}\n\n'
            f"Equation: {item.get('text', item.get('equation', 'N/A'))}\n"
            f"Format: {item.get('equation_format', 'latex')}\n"
        )
        if context:
            prompt += f"\nContext:\n{context}\n"
        return call_llm(prompt, system_prompt="You are an expert mathematician.")

    else:
        prompt = (
            f"Analyze this {item_type} content:\n"
            f"{json.dumps(item, indent=2, default=str)}\n"
            f"Provide a concise description in 2-3 sentences."
        )
        return call_llm(prompt, system_prompt="You are an expert content analyst.")


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


def _format_multimodal_chunk(item: dict, *, use_llm: bool = False) -> Tuple[str, str, str]:
    """Return (content, entity_base, original_type) for a multimodal item.

    *entity_base* is the type category (e.g. "image", "table") used to build
    the final entity name like "Image_1".  Uses the same templates as
    raganything/prompt.py.
    """
    item_type = item.get("type", "unknown")
    if use_llm:
        enhanced_caption = enhance_caption(item, item_type)
    else:
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
    use_llm: bool = False,
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
        content, entity_base, original_type = _format_multimodal_chunk(item, use_llm=use_llm)
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
    text_count = mm_count = 0
    types: dict[str, int] = {}
    for c in chunks:
        if c.get("is_multimodal"):
            mm_count += 1
            t = c.get("original_type", "unknown")
            types[t] = types.get(t, 0) + 1
        else:
            text_count += 1
    logger.info("--- Summary ---")
    logger.info("Source file:       %s", file_path)
    logger.info("Total chunks:      %d", len(chunks))
    logger.info("  Text chunks:     %d", text_count)
    logger.info("  Multimodal:      %d", mm_count)
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

        pdf = next(tmp_path.glob("*.pdf"), None)
        if pdf is None:
            raise RuntimeError(
                f"No PDF generated for {source_path.name}"
            )
        if pdf.stat().st_size < 100:
            raise RuntimeError("Generated PDF appears empty or corrupt.")

        final = output_dir / f"{source_path.stem}.pdf"
        shutil.copy2(pdf, final)

    return final
