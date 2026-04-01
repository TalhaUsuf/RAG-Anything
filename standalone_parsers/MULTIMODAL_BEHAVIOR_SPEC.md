# Multimodal Document Processing Behavior Spec

## Source: RAG-Anything `raganything/` original implementation

This spec codifies the exact processing behavior for multimodal documents
as implemented in the original RAG-Anything codebase. Any standalone parser
must replicate this behavior.

---

## 1. Content Types & Their Fields

MinerU extracts documents into a `content_list: List[Dict]` with these types:

### Text Block
```python
{"type": "text", "text": str, "page_idx": int, "text_level": int}
```
- `text_level`: 0 = body, 1+ = header level

### Image Block
```python
{
    "type": "image",
    "img_path": str,              # absolute path to extracted image file
    "image_caption": list[str],   # captions (alias: img_caption)
    "image_footnote": list[str],  # footnotes (alias: img_footnote)
    "page_idx": int
}
```

### Table Block
```python
{
    "type": "table",
    "img_path": str,              # optional table screenshot path
    "table_body": str,            # full markdown table content
    "table_caption": list[str],
    "table_footnote": list[str],
    "page_idx": int
}
```

### Equation Block
```python
{
    "type": "equation",
    "text": str,                  # LaTeX string
    "text_format": str,           # "latex" or other
    "page_idx": int
}
```

### List Block (MinerU-specific)
```python
{
    "type": "list",
    "sub_type": str,              # "text" etc.
    "list_items": list[str],
    "page_idx": int
}
```

---

## 2. Processing Pipeline

```
Document
  → MinerU parse → content_list
  → separate_content() → (text_string, multimodal_items)
  → Text path:  text → chunk by ~1200 chars → text chunks
  → Multimodal path: for each item:
      1. Select processor by type (image→VisionLLM, table/equation/generic→TextLLM)
      2. Extract surrounding context (±1 page window, text-only items)
      3. Call LLM with type-specific prompt + content + context
      4. Parse JSON response → (detailed_description, entity_info)
      5. Format chunk using type-specific template
      6. Store as multimodal chunk
```

---

## 3. LLM Selection Per Content Type

| Type     | LLM Used      | Input Format                    |
|----------|---------------|---------------------------------|
| image    | Vision LLM    | base64 image + vision prompt    |
| table    | Text LLM      | table_body markdown + prompt    |
| equation | Text LLM      | LaTeX text + prompt             |
| list     | Text LLM      | list items JSON + generic prompt|
| generic  | Text LLM      | content JSON + generic prompt   |

**Critical**: Images use the VISION model with base64-encoded image data.
Tables, equations, and all other types use the TEXT model only.

---

## 4. Prompt Templates

### Image (Vision LLM)
System: "You are an expert image analyst. Provide detailed, accurate descriptions."
User prompt (with context):
```
Please analyze this image in detail, considering the surrounding context.
Provide a JSON response with:
{
    "detailed_description": "comprehensive visual description...",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "max 100 words"
    }
}
Context from surrounding content: {context}
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}
```

### Table (Text LLM)
System: "You are an expert data analyst. Provide detailed table analysis with specific insights."
User prompt:
```
Analyze this table content...
{
    "detailed_description": "structure, headers, data patterns, statistical insights...",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "max 100 words"
    }
}
Table body: {table_body}
Caption: {table_caption}
Footnotes: {table_footnote}
```

### Equation (Text LLM)
System: "You are an expert mathematician. Provide detailed mathematical analysis."
User prompt:
```
Analyze this equation...
{
    "detailed_description": "meaning, variables, operations, applications...",
    "entity_info": {...}
}
Equation: {equation_text}
Format: {equation_format}
```

### Generic (Text LLM)
System: "You are an expert content analyst specializing in {content_type} content."
Same JSON structure, content passed as-is.

---

## 5. Expected LLM Response Format

All content types return JSON:
```json
{
    "detailed_description": "string — comprehensive analysis",
    "entity_info": {
        "entity_name": "string — descriptive name for the content",
        "entity_type": "string — image|table|equation|etc",
        "summary": "string — max 100 words"
    }
}
```

Robust parsing required: try direct JSON, cleanup, regex extraction as fallbacks.

---

## 6. Chunk Templates

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

---

## 7. Context Extraction

For each multimodal item:
- Window: ±1 page (configurable)
- Filter: text-only items from surrounding pages
- Format: "[Page N] text content..." joined
- Truncate to max 2000 tokens
- Passed as `{context}` in prompt templates

---

## 8. Final Chunk Dict Structure

### Text Chunk
```python
{
    "content": str,            # raw text segment
    "tokens": int,             # word count
    "full_doc_id": str,        # "doc-{md5}"
    "chunk_order_index": int,  # position (text first)
    "file_path": str,          # source filename
    "is_multimodal": False,
}
```

### Multimodal Chunk
```python
{
    "content": str,              # template-formatted with LLM description
    "tokens": int,
    "full_doc_id": str,
    "chunk_order_index": int,    # after all text chunks
    "file_path": str,
    "is_multimodal": True,
    "modal_entity_name": str,    # "EntityName (type)"
    "original_type": str,        # image|table|equation|list|generic
    "page_idx": int,
}
```

---

## 9. Markdown-Specific Requirements

For `.md` files containing image references `![alt](path/to/image.png)`:

1. **Detect** image references using regex: `!\[([^\]]*)\]\(([^)]+)\)`
2. **Resolve** image path relative to the markdown file's directory
3. **Validate** the image file exists and is readable
4. **Create** an image content block:
   ```python
   {
       "type": "image",
       "img_path": "/absolute/path/to/image.png",
       "image_caption": ["alt text from markdown"],
       "image_footnote": [],
       "page_idx": 0
   }
   ```
5. **Remove** the image reference line from text content
6. **Process** the image through Vision LLM (base64 encode → send with vision prompt)
7. **Create** multimodal chunk with Vision LLM description

The text content (with image refs removed) goes through the normal
text → PDF → MinerU → text chunks path.

---

## 10. Vision LLM Call Spec

For images, the Vision LLM is called with an OpenAI-compatible multimodal
message format:

```python
{
    "model": "vision-model-name",
    "messages": [
        {"role": "system", "content": "You are an expert image analyst..."},
        {"role": "user", "content": [
            {"type": "text", "text": "vision prompt with context..."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_data}"
            }}
        ]}
    ],
    "max_tokens": 1024,
    "temperature": 0.1
}
```

The response follows standard OpenAI chat completion format:
`response["choices"][0]["message"]["content"]` → JSON string to parse.
