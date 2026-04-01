---
description: "Scaffold a new modal processor for RAG-Anything following the BaseModalProcessor pattern"
---

# Add New Modal Processor to RAG-Anything

Scaffold and implement a new modal content processor by following the pattern in `raganything/modalprocessors.py`.

## Modal Processor Architecture

All processors extend `BaseModalProcessor` and must implement `generate_description_only()`:

```python
class BaseModalProcessor:
    def __init__(self, lightrag: LightRAG, modal_caption_func, context_extractor: ContextExtractor = None)

    # Abstract — must implement:
    async def generate_description_only(
        self, modal_content, content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]

    # Inherited — use as-is:
    async def _create_entity_and_chunk(modal_chunk, entity_info, file_path, ...)
    def _robust_json_parse(response: str) -> dict
    def _get_context_for_item(item_info) -> str
    def set_content_source(content_source, content_format="auto")
    async def _process_chunk_for_extraction(chunk_id, modal_entity_name, batch_mode)
```

## Required Response Format

The LLM/vision model must return JSON matching:
```json
{
    "detailed_description": "comprehensive analysis text",
    "entity_info": {
        "entity_name": "descriptive name",
        "entity_type": "your_content_type",
        "summary": "concise summary, max 100 words"
    }
}
```

## Implementation Steps

When the user provides the content type as $ARGUMENTS, implement:

### Step 1: Add Prompt Templates (`raganything/prompt.py`)

Add to the PROMPTS dictionary:
- `YOUR_TYPE_ANALYSIS_SYSTEM`: System prompt for the analysis
- `your_type_prompt`: Analysis prompt without context
- `your_type_prompt_with_context`: Analysis prompt with surrounding context
- `your_type_chunk`: Template for creating the stored chunk text

Follow the existing pattern from image/table/equation prompts.

### Step 2: Create Processor Class (`raganything/modalprocessors.py`)

```python
class YourTypeModalProcessor(BaseModalProcessor):
    def __init__(self, lightrag, modal_caption_func, context_extractor=None):
        super().__init__(lightrag, modal_caption_func, context_extractor)

    async def generate_description_only(self, modal_content, content_type, item_info=None, entity_name=None):
        # 1. Parse modal_content (JSON string or plain text)
        # 2. Extract relevant fields
        # 3. Get context via self._get_context_for_item(item_info)
        # 4. Select prompt (with_context or base)
        # 5. Call self.modal_caption_func with prompt + system prompt
        # 6. Parse response via self._robust_json_parse()
        # 7. Return (description, entity_info)

    async def process_multimodal_content(self, modal_content, content_type, file_path="manual",
                                          entity_name=None, item_info=None, batch_mode=False,
                                          doc_id=None, chunk_order_index=0):
        # 1. Call generate_description_only()
        # 2. Build modal_chunk from PROMPTS["your_type_chunk"] template
        # 3. Call self._create_entity_and_chunk()
        # 4. Return result

    def _parse_response(self, response, entity_name=None):
        # 1. Use self._robust_json_parse(response)
        # 2. Extract detailed_description and entity_info
        # 3. Validate required fields (entity_name, entity_type, summary)
        # 4. Append entity_type suffix to entity_name
        # 5. Return (description, entity_info) or fallback
```

### Step 3: Register Processor

**In `raganything/utils.py`** — Update `get_processor_for_type()`:
```python
elif content_type == "your_type":
    return modal_processors.get("your_type")
```

**In `raganything/raganything.py`** — Add to `_initialize_processors()`:
```python
if self.config.enable_your_type_processing:
    self.modal_processors["your_type"] = YourTypeModalProcessor(
        lightrag=self.lightrag,
        modal_caption_func=caption_func,
        context_extractor=self.context_extractor,
    )
```

**In `raganything/config.py`** — Add config field:
```python
enable_your_type_processing: bool = field(default=True)
```

### Step 4: Update Content Separation

**In `raganything/utils.py`** — Update `separate_content()` if needed to recognize the new type.

## Implementation Checklist

1. Read `raganything/modalprocessors.py` to study existing processor patterns
2. Read `raganything/prompt.py` for prompt template patterns
3. Add prompt templates to prompt.py
4. Create the processor class in modalprocessors.py
5. Add `_parse_response()` method with robust JSON parsing fallbacks
6. Register in utils.py `get_processor_for_type()`
7. Register in raganything.py `_initialize_processors()`
8. Add config toggle in config.py
9. Update content separation in utils.py if needed
10. Test with a sample document containing the new content type

$ARGUMENTS
