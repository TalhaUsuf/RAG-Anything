---
description: "Scaffold a new document parser for RAG-Anything following the established Parser base class pattern"
---

# Add New Parser to RAG-Anything

Scaffold and implement a new document parser by following the established pattern in `raganything/parser.py`.

## Parser Architecture

All parsers extend the `Parser` base class and must implement 4 abstract methods:

```python
class Parser:
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # Abstract methods to implement:
    def parse_pdf(self, pdf_path, output_dir=None, method="auto", lang=None, **kwargs) -> List[Dict[str, Any]]
    def parse_image(self, image_path, output_dir=None, lang=None, **kwargs) -> List[Dict[str, Any]]
    def parse_document(self, file_path, method="auto", output_dir=None, lang=None, **kwargs) -> List[Dict[str, Any]]
    def check_installation(self) -> bool

    # Inherited helper methods (do not override unless necessary):
    def _unique_output_dir(output_dir, file_path) -> Path
    def convert_office_to_pdf(cls, office_path, output_dir) -> Path
    def convert_text_to_pdf(cls, text_path, output_dir) -> Path
```

## Required Output Format

All `parse_*` methods must return `List[Dict[str, Any]]` with these block types:

```python
# Text block
{"type": "text", "text": "content string", "page_idx": 0}

# Image block
{"type": "image", "img_path": "/absolute/path/to/image.png",
 "image_caption": "caption text", "image_footnote": "footnote text", "page_idx": 0}

# Table block
{"type": "table", "table_body": [["header1", "header2"], ["val1", "val2"]],
 "table_caption": "caption", "table_footnote": "footnote",
 "img_path": "/absolute/path/to/table_image.png", "page_idx": 0}

# Equation block
{"type": "equation", "text": "E = mc^2", "text_format": "latex", "page_idx": 0}
```

## Registration Steps

After creating the parser class, register it in these locations:

1. **parser.py** — Add to `SUPPORTED_PARSERS` tuple and `get_parser()` factory:
   ```python
   SUPPORTED_PARSERS = ("mineru", "docling", "paddleocr", "yournewparser")

   def get_parser(parser_type: str) -> Parser:
       # ... existing parsers ...
       if parser_name == "yournewparser":
           return YourNewParser()
   ```

2. **config.py** — The parser name is already configurable via `config.parser` field

3. **env.example** — Document the new parser option

## Implementation Checklist

When the user provides the parser name as $ARGUMENTS, do the following:

1. Read `raganything/parser.py` to understand the current patterns
2. Create the new parser class at the end of parser.py (before `get_parser()`)
3. Implement all 4 abstract methods following existing parser patterns:
   - `parse_pdf()`: Handle PDF input, return content list
   - `parse_image()`: Handle image input, return content list
   - `parse_document()`: Route by file extension to appropriate method
   - `check_installation()`: Verify dependencies are available
4. Add the parser to `SUPPORTED_PARSERS` tuple
5. Add the parser to `get_parser()` factory function
6. Use `_unique_output_dir()` for collision-proof output directories
7. Convert all image paths to absolute paths in the output
8. Validate paths against directory traversal (use `os.path.realpath()`)
9. Add appropriate logging throughout
10. Update `env.example` to document the new parser option

## Security Requirements
- Validate all file paths against directory traversal attacks
- Use `os.path.realpath()` to resolve symlinks before path checks
- Never execute user-provided strings as commands without sanitization
- Use subprocess with explicit argument lists, never shell=True

## Error Handling Pattern
- Raise `FileNotFoundError` for missing input files
- Raise `ValueError` for unsupported file formats
- Use `self.logger.warning()` for recoverable issues
- Use `self.logger.error()` for failures with fallback behavior

$ARGUMENTS
