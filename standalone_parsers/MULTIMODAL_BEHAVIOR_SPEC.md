# Multimodal Document Processing Behavior Spec

## Source

Derived from deep analysis of the original RAG-Anything codebase by a
background agent that traced every function call, prompt template, and
data structure in `raganything/processor.py`, `raganything/modalprocessors.py`,
`raganything/prompt.py`, `raganything/utils.py`, and `raganything/parser.py`.

## Service Endpoints

All endpoints are configured via the `.env` file (see `.env.example`):

| Env Variable | Service | Example Value |
|-------------|---------|---------------|
| `MINERU_API_URL` | Remote MinerU | `http://host:40050` |
| `LLM_BASE_URL` / `LLM_MODEL` | Text LLM | OpenAI-compatible `/v1` endpoint |
| `VLM_BASE_URL` / `VLM_MODEL` | Vision LLM | OpenAI-compatible multimodal endpoint |
| `EMBED_BASE_URL` / `EMBED_MODEL` | Embeddings | OpenAI-compatible embedding endpoint |

All are OpenAI-compatible HTTP APIs. No URLs are hardcoded in source code.

---

## 1. Content Types from MinerU

MinerU extracts documents into `content_list: List[Dict]`:

### Text Block
```python
{"type": "text", "text": str, "page_idx": int, "text_level": int}
# text_level: 0 = body, 1+ = header level
```

### Image Block
```python
{
    "type": "image",
    "img_path": str,                  # absolute path to extracted image file
    "image_caption": list[str],       # captions (alias: img_caption for MinerU 1.x)
    "image_footnote": list[str],      # footnotes (alias: img_footnote for MinerU 1.x)
    "page_idx": int,
    "bbox": [x1, y1, x2, y2]         # optional bounding box
}
```

### Table Block
```python
{
    "type": "table",
    "img_path": str,                  # optional table screenshot path
    "table_body": str,                # full markdown table content (can be very long)
    "table_caption": list[str],
    "table_footnote": list[str],
    "page_idx": int
}
```

### Equation Block
```python
{
    "type": "equation",
    "text": str,                      # LaTeX string
    "text_format": str,               # "latex" or other
    "page_idx": int
}
```

### List Block (MinerU-specific)
```python
{
    "type": "list",
    "sub_type": str,
    "list_items": list[str],
    "page_idx": int,
    "bbox": [x1, y1, x2, y2]
}
```

### Field Normalization (parser.py lines 883-900)
- `img_caption` ↔ `image_caption` (bidirectional copy)
- `img_footnote` ↔ `image_footnote` (bidirectional copy)
- All relative paths in `img_path`, `table_img_path`, `equation_img_path`
  are converted to absolute paths with traversal check.

---

## 2. Processing Pipeline — 7-Stage Batch (processor.py lines 706-881)

```
Document
  → MinerU parse → content_list
  → separate_content() → (text_string, multimodal_items)
  → set_content_source_for_context(content_list)  ← provides context to processors
  → Text path:  text → LightRAG.ainsert() → text chunks
  → Multimodal path: _process_multimodal_content_batch_type_aware()
      STAGE 1: Concurrent generate_description_only() per item
               (controlled by asyncio.Semaphore(max_parallel_insert=2))
      STAGE 2: _convert_to_lightrag_chunks_type_aware() → apply chunk templates
      STAGE 3: _store_chunks_to_lightrag_storage_type_aware() → text_chunks + chunks_vdb
      STAGE 3.5: _store_multimodal_main_entities() → entities_vdb
      STAGE 4: _batch_extract_entities_lightrag_style_type_aware() → entity extraction
      STAGE 5: _batch_add_belongs_to_relations_type_aware() → "belongs_to" edges
      STAGE 6: _batch_merge_lightrag_style_type_aware() → merge into knowledge graph
      STAGE 7: _update_doc_status_with_chunks_type_aware() → update doc status
```

Fallback: If batch fails, falls back to MODE B individual processing
(processor.py lines 551-700) which calls `processor.process_multimodal_content()`
sequentially per item.

---

## 3. LLM Routing Per Content Type (raganything.py lines 175-217)

| Type     | LLM Used                                | Input Format                           |
|----------|-----------------------------------------|----------------------------------------|
| image    | `vision_model_func` OR `llm_model_func` | base64 image + vision prompt           |
| table    | `llm_model_func` only                   | table_body markdown + table prompt     |
| equation | `llm_model_func` only                   | LaTeX text + equation prompt           |
| list     | `llm_model_func` only                   | list items JSON + generic prompt       |
| generic  | `llm_model_func` only                   | content JSON + generic prompt          |

**Critical**: Images use the VISION model (`InternVL3-38B`) with base64-encoded
image data via OpenAI multimodal message format. All other types use the TEXT
model (`llama-3.1-70b`).

---

## 4. Context Extraction (modalprocessors.py lines 33-357)

### ContextConfig Dataclass
```python
@dataclass
class ContextConfig:
    context_window: int = 1           # ±N pages/chunks around current item
    context_mode: str = "page"        # "page" | "chunk" | "token"
    max_context_tokens: int = 2000    # hard limit on context size
    include_headers: bool = True      # include header text with markers
    include_captions: bool = True     # include [Image: caption] / [Table: caption]
    filter_content_types: list = None # defaults to ["text"]
```

### Page Mode (default)
```python
start_page = max(0, current_page - context_window)
end_page = current_page + context_window + 1
```
- Collects text-type items from pages in `[start_page, end_page)`
- Adds `[Page N]` markers when items cross page boundaries
- Extracts text via `_extract_text_from_item()`:
  - text items: raw text (with `## Header` markers if text_level > 0)
  - image items: `[Image: caption1, caption2]`
  - table items: `[Table: caption1, caption2]`

### Chunk Mode
Uses item index instead of page_idx:
```python
start_idx = max(0, current_index - context_window)
end_idx = min(len(content_list), current_index + context_window + 1)
```
Excludes the current item itself.

### Truncation
Truncates to `max_context_tokens` tokens if tokenizer available,
otherwise falls back to character-based truncation. Tries to end
at sentence boundary (`.` or `\n`).

---

## 5. Exact Prompt Templates (from raganything/prompt.py)

### 5.1 Image — Vision LLM

**System prompt** (line 15):
```
You are an expert image analyst. Provide detailed, accurate descriptions.
```

**User prompt WITHOUT context** (lines 32-57):
```
Please analyze this image in detail and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Always use specific names instead of pronouns",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance (max 100 words)"
    }
}

Additional context:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval.
```

**User prompt WITH context** (lines 60-89):
Same structure, with added:
```
Context from surrounding content:
{context}
```
And guidelines extended with:
```
    - Explain relationships between elements and how they relate to the surrounding context
    - Reference connections to the surrounding content when relevant
```

### 5.2 Table — Text LLM

**System prompt** (line 21):
```
You are an expert data analyst. Provide detailed table analysis with specific insights.
```

**User prompt** (lines 101-127):
```
Please analyze this table content and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }
}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data.
```

**Long table handling**: full `table_body` is sent to LLM without truncation.

### 5.3 Equation — Text LLM

**System prompt** (line 24):
```
You are an expert mathematician. Provide detailed mathematical analysis.
```

**User prompt** (lines 163-188):
```
Please analyze this mathematical equation and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }
}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance.
```

### 5.4 Generic — Text LLM

**System prompt** (line 27):
```
You are an expert content analyst specializing in {content_type} content.
```

**User prompt** (lines 223-272): Same JSON structure, content passed as-is.

---

## 6. Expected LLM Response Format

All content types must return JSON:
```json
{
    "detailed_description": "string — comprehensive analysis",
    "entity_info": {
        "entity_name": "string — descriptive name (e.g. 'Market Growth Chart')",
        "entity_type": "string — image|table|equation|etc",
        "summary": "string — max 100 words"
    }
}
```

### Robust Parsing (modalprocessors.py lines 547-693)

`_robust_json_parse()` uses 4-strategy fallback:
1. **Direct parse**: `json.loads(response)`
2. **Basic cleanup**: fix smart quotes, trailing commas → parse
3. **Progressive quote fix**: fix unescaped backslashes/quotes → parse
4. **Regex extraction**: extract `detailed_description` and `entity_info` fields
   with regex as last resort

`_extract_all_json_candidates()` handles reasoning models (strips thinking
tags for Qwen2.5-think, DeepSeek-R1, etc.).

After parsing, `entity_type` is appended to `entity_name` as suffix:
`"Market Growth Chart"` → `"Market Growth Chart (image)"`

---

## 7. Chunk Templates (prompt.py lines 274-300)

### Image Chunk
```
Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}
```

### Table Chunk
```
Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}
```

### Equation Chunk
```
Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}
```

### Generic Chunk
```
{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}
```

`{enhanced_caption}` = the `detailed_description` from the LLM response.

---

## 8. Vision LLM Call Format (OpenAI-compatible multimodal)

```python
{
    "model": "OpenGVLab/InternVL3-38B",
    "messages": [
        {"role": "system", "content": "You are an expert image analyst..."},
        {"role": "user", "content": [
            {"type": "text", "text": "<vision prompt with context>"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,{base64_data}"
            }}
        ]}
    ],
    "max_tokens": 1024,
    "temperature": 0.1
}
```

The `modal_caption_func` signature in the original code:
```python
async def modal_caption_func(
    prompt: str,
    image_data: str = None,      # base64-encoded image
    system_prompt: str = None,
) -> str
```

---

## 9. Final Chunk Dict Structure

### Text Chunk
```python
{
    "content": str,               # raw text segment (~1200 chars)
    "tokens": int,                # token count
    "full_doc_id": str,           # "doc-{md5hash}"
    "chunk_order_index": int,     # position (text chunks first, then multimodal)
    "file_path": str,             # source filename
    "is_multimodal": False,
}
```

### Multimodal Chunk
```python
{
    "content": str,               # template-formatted with LLM description
    "tokens": int,
    "full_doc_id": str,
    "chunk_order_index": int,     # after all text chunks
    "file_path": str,
    "is_multimodal": True,
    "modal_entity_name": str,     # "EntityName (type)" e.g. "Market Chart (image)"
    "original_type": str,         # image|table|equation|list|generic
    "page_idx": int,
}
```

---

## 10. Markdown-Specific Requirements

For `.md` files containing `![alt](path/to/image.png)`:

1. **Detect** image references: regex `!\[([^\]]*)\]\(([^)]+)\)`
2. **Resolve** path relative to the markdown file's directory
3. **Validate** image file exists and is readable
4. **Create** image content block matching MinerU format:
   ```python
   {"type": "image", "img_path": str, "image_caption": [alt], "image_footnote": [], "page_idx": 0}
   ```
5. **Remove** image reference lines from text before PDF conversion
6. **Extract context** from surrounding text (±1 page window)
7. **Encode** image to base64 and send to Vision LLM with `vision_prompt_with_context`
8. **Parse** JSON response with robust 4-strategy fallback
9. **Format** chunk using `image_chunk` template with `enhanced_caption` = `detailed_description`
10. **Merge** image chunks with MinerU text/list chunks in final output

---

## 11. Entity Naming Convention

Entity names follow the pattern from `_parse_response()`:
- LLM suggests `entity_name` in the JSON response
- Code appends `(entity_type)` suffix: `"Market Growth Chart"` → `"Market Growth Chart (image)"`
- Counter appended for uniqueness: `"Image_1"`, `"Table_2"`, etc.
- If LLM doesn't provide entity_name, hash-based fallback: `"image_{md5[:8]}"`
