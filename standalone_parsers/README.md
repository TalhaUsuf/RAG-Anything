# Standalone Parsers for RAG-Anything

## Overview

The standalone parsers are self-contained command-line scripts that convert
documents of various formats into chunked JSON suitable for retrieval-augmented
generation (RAG) pipelines.  Each parser converts its input to PDF (or an
image format MinerU accepts), runs it through MinerU for layout-aware content
extraction, and emits a list of structured chunks on stdout.

The parsers share a common base class (`BaseParser`) that implements the full
pipeline as a Template Method.  Concrete parsers only need to implement a
`preprocess()` step that turns the source file into something MinerU can
process.

---

## Prerequisites

| Requirement | Why |
|---|---|
| **Python 3.10+** | Type-union syntax (`str \| None`) used throughout |
| **MinerU** (local install) | PDF/image content extraction in local mode |
| **Remote MinerU API** | Alternative to a local install; set `MINERU_API_URL` in `.env` |
| **LibreOffice** (`soffice` on PATH) | Converts Office and HTML files to PDF |
| **Pillow** (`pip install Pillow`) | Converts non-native image formats (BMP, TIFF, GIF, WebP) to PNG |
| **ReportLab** (`pip install reportlab`) | Converts plain-text and Markdown files to PDF; HTML fallback |

Install the Python dependencies:

```bash
pip install Pillow reportlab
```

LibreOffice must be available as `soffice` on the system PATH.  On
Debian/Ubuntu:

```bash
sudo apt-get install libreoffice
```

---

## Configuration

Copy the example environment file and fill in the service URLs:

```bash
cp .env.example .env
```

The `.env` file controls five service endpoints:

| Variable | Purpose | Example |
|---|---|---|
| `MINERU_API_URL` | Remote MinerU API base URL | `http://69.48.159.8:40050` |
| `LLM_BASE_URL` | OpenAI-compatible text LLM endpoint | `http://host:30000/v1` |
| `LLM_MODEL` | Text LLM model name | `llama-3.1-70b` |
| `LLM_API_KEY` | API key for the text LLM (optional) | |
| `VLM_BASE_URL` | OpenAI-compatible vision LLM endpoint | `http://host:23333/v1` |
| `VLM_MODEL` | Vision LLM model name | `OpenGVLab/InternVL3-38B` |
| `VLM_API_KEY` | API key for the vision LLM (optional) | |
| `EMBED_BASE_URL` | Embedding model endpoint | `http://host:30007/v1` |
| `EMBED_MODEL` | Embedding model name | `Nexus_Embedding_Model_seq_8192_embd_1024` |
| `EMBED_API_KEY` | API key for the embedding model (optional) | |

Only `MINERU_API_URL` is required when using `--remote`.  The `LLM_*` and
`VLM_*` variables are only needed when using `--use-llm` for enhanced
multimodal captions.

---

## Supported File Types

| Script | Extensions | Pre-processing |
|---|---|---|
| `parser_images.py` | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.webp` | Non-native formats (BMP, TIFF, GIF, WebP) converted to PNG via Pillow |
| `parser_office.py` | `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx` | Converted to PDF via LibreOffice |
| `parser_text.py` | `.txt`, `.md` | Converted to PDF via ReportLab; Markdown images extracted first |
| `parser_html.py` | `.html`, `.htm`, `.xhtml` | Converted to PDF via LibreOffice (ReportLab text fallback on failure) |

**Total: 18 extensions** across four parsers.

---

## Usage

All four parsers share the same CLI interface, defined in `BaseParser.run_cli()`:

```
python <parser>.py <file> [output_dir] [--remote] [--use-llm] [--chunk-size N] [-l LANG] [-b BACKEND] [-d DEVICE]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `file` | Yes | -- | Path to the input file |
| `output_dir` | No | `./output` | Directory for intermediate files and MinerU output |
| `--remote` | No | off | Use the remote MinerU API instead of the local CLI |
| `--use-llm` | No | off | Generate enhanced captions for multimodal items via LLM/VLM |
| `--chunk-size N` | No | `1200` | Target chunk size in characters |
| `-l`, `--lang` | No | `None` | Language hint passed to MinerU (e.g. `en`, `zh`) |
| `-b`, `--backend` | No | `None` | MinerU backend (local mode uses MinerU default; remote defaults to `hybrid-auto-engine`) |
| `-d`, `--device` | No | `None` | Device for local MinerU (`cpu` or `cuda`) |

### Parser-specific commands

**Images** -- parse image files via MinerU OCR:

```bash
python parser_images.py <file> [output_dir] [--remote] [--use-llm] [--chunk-size N] [-l LANG] [-b BACKEND] [-d DEVICE]
```

**Office documents** -- convert via LibreOffice, then parse with MinerU:

```bash
python parser_office.py <file> [output_dir] [--remote] [--use-llm] [--chunk-size N] [-l LANG] [-b BACKEND] [-d DEVICE]
```

**Text / Markdown** -- convert via ReportLab, then parse with MinerU:

```bash
python parser_text.py <file> [output_dir] [--remote] [--use-llm] [--chunk-size N] [-l LANG] [-b BACKEND] [-d DEVICE]
```

**HTML** -- convert via LibreOffice (or ReportLab fallback), then parse with MinerU:

```bash
python parser_html.py <file> [output_dir] [--remote] [--use-llm] [--chunk-size N] [-l LANG] [-b BACKEND] [-d DEVICE]
```

---

## Examples

All examples assume you are running from the `standalone_parsers/` directory.

### Parse a PNG image (local MinerU)

```bash
python parser_images.py test_samples/sample_image.png ./output
```

### Parse a BMP image (remote MinerU, with LLM captions)

```bash
python parser_images.py test_samples/sample_image.bmp ./output --remote --use-llm
```

### Parse a Markdown file with embedded images

```bash
python parser_text.py test_samples/sample_multimodal.md ./output --use-llm
```

### Parse a plain text file with a custom chunk size

```bash
python parser_text.py test_samples/sample_text.txt ./output --chunk-size 800
```

### Parse an HTML document

```bash
python parser_html.py test_samples/sample_document.html ./output --remote
```

### Parse an Office document with language hint

```bash
python parser_office.py report.docx ./output -l en -d cuda
```

### Redirect JSON output to a file

Every parser prints the chunk list as JSON to stdout.  Log messages go to
stderr, so you can capture just the data:

```bash
python parser_images.py test_samples/chart.png ./output > chunks.json
```

---

## Output Format

Each parser prints a JSON array of chunk objects to stdout.  There are two
kinds of chunks: **text chunks** and **multimodal chunks**.

### Text chunk

```json
{
  "content": "The extracted text content of this chunk...",
  "tokens": 142,
  "full_doc_id": "doc-a1b2c3d4e5f6",
  "chunk_order_index": 0,
  "file_path": "sample_image.png",
  "is_multimodal": false
}
```

### Multimodal chunk (image, table, or equation)

```json
{
  "content": "[Image]\nImage path: /path/to/image.png\nCaptions: ...\nFootnotes: ...\nEnhanced caption: ...",
  "tokens": 87,
  "full_doc_id": "doc-a1b2c3d4e5f6",
  "chunk_order_index": 3,
  "file_path": "sample_image.png",
  "is_multimodal": true,
  "modal_entity_name": "Image_1",
  "original_type": "image",
  "page_idx": 0
}
```

| Field | Type | Description |
|---|---|---|
| `content` | string | The chunk text.  For multimodal items this includes metadata and captions. |
| `tokens` | int | Approximate word count (`len(content.split())`). |
| `full_doc_id` | string | Deterministic hash ID for the entire document. |
| `chunk_order_index` | int | Zero-based position of this chunk in the document. |
| `file_path` | string | Base name of the source file. |
| `is_multimodal` | bool | `false` for text chunks, `true` for images/tables/equations. |
| `modal_entity_name` | string | (Multimodal only) Entity identifier such as `Image_1`, `Table_2`, `Equation_1`. |
| `original_type` | string | (Multimodal only) One of `image`, `table`, `equation`, or a custom type. |
| `page_idx` | int | (Multimodal only) Zero-based page index where the item was found. |

Multimodal content templates vary by type:

- **Image** -- image path, captions, footnotes, enhanced caption.
- **Table** -- table image path, table caption, table body (HTML/text), table footnote, enhanced caption.
- **Equation** -- equation text (LaTeX), equation format, enhanced caption.

When `--use-llm` is not set, the `enhanced_caption` field reads
`[Requires LLM analysis]`.

---

## Architecture

The parsers follow the **Template Method** pattern combined with a **Strategy**
facade for external services.

### BaseParser (Template Method)

`base_parser.py` defines the abstract `BaseParser` class.  Its `parse()`
method drives the full pipeline:

1. **Validate** -- check that the file exists and has a supported extension.
2. **Set up output directory** -- create a unique subdirectory to avoid collisions.
3. **Preprocess** (abstract) -- convert the input to a format MinerU accepts.
4. **Run MinerU** -- execute the local CLI or call the remote API.
5. **Read MinerU output** -- load the structured content list.
6. **Separate content** -- split into full text and multimodal items.
7. **Merge extras** -- combine any items extracted during preprocessing (e.g. Markdown images).
8. **Build chunks** -- split text into sized segments, format multimodal items, optionally enhance captions via LLM.
9. **Return** -- emit the chunk list.

Subclasses only implement two methods:

- `supported_formats()` -- returns the set of accepted file extensions.
- `preprocess(file_path, out_dir)` -- converts the input file and returns `(mineru_input_path, extra_multimodal_items)`.

### ParserConfig

A dataclass holding all tuneable parameters (output directory, remote mode,
LLM flag, chunk size, language, backend, device).  Populated automatically
from CLI arguments by `run_cli()`.

### AIServices

A facade that wraps all external service calls (`run_mineru`,
`run_mineru_remote`, `call_llm`, `call_vlm`, `enhance_caption`) so that
parsers never import `_common.py` helpers directly.  This makes it
straightforward to mock services in tests.

### Class hierarchy

```
BaseParser (abstract)
  |-- ImageParser   (parser_images.py)  -- Pillow conversion + OCR mode
  |-- OfficeParser  (parser_office.py)  -- LibreOffice PDF conversion
  |-- TextParser    (parser_text.py)    -- ReportLab PDF + Markdown image extraction
  |-- HtmlParser    (parser_html.py)    -- LibreOffice PDF (ReportLab fallback)
```

---

## Legacy Scripts

The original single-file parser scripts still exist for backward compatibility:

| Legacy script | Replacement |
|---|---|
| `parse_images.py` | `parser_images.py` |
| `parse_office.py` | `parser_office.py` |
| `parse_text.py` | `parser_text.py` |
| `parse_html.py` | `parser_html.py` |

The legacy scripts are standalone and do not use `BaseParser`.  New work
should use the `parser_*.py` versions, which share the common pipeline,
support all CLI flags uniformly, and are easier to extend.
