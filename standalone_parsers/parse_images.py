#!/usr/bin/env python3
"""
Standalone image file parser for RAG-Anything.

Processes image files (.png, .jpg, .jpeg, .bmp, .tiff, .tif, .gif, .webp)
through MinerU OCR pipeline and produces chunk JSON.

No imports from the raganything package.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from _common import (
    build_chunks,
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

IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
MINERU_NATIVE_FORMATS = {".png", ".jpeg", ".jpg"}


# ---------------------------------------------------------------------------
# Image format conversion
# ---------------------------------------------------------------------------

def convert_to_png(image_path: str, output_dir: str) -> Path:
    """Convert non-native image formats to PNG for MinerU compatibility."""
    from PIL import Image

    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s to PNG ...", image_path.name)

    img = Image.open(image_path)

    if img.mode in ("RGBA", "LA", "P"):
        if img.mode == "P":
            img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    png_path = output_dir / f"{image_path.stem}.png"
    img.save(str(png_path), format="PNG")
    logger.info("Converted image saved to %s", png_path)
    return png_path


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
        "output_dir", nargs="?", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument("-l", "--lang", default=None, help="Language hint for MinerU OCR")
    parser.add_argument("-b", "--backend", default=None, help="MinerU backend to use")
    parser.add_argument("-d", "--device", default=None, help="Device for MinerU (e.g. cpu, cuda)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Target text chunk size in chars (default: 1200)")
    parser.add_argument("--remote", action="store_true", help="Use remote MinerU API instead of local CLI")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for enhanced captions")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    suffix = file_path.suffix.lower()
    if suffix not in IMAGE_FORMATS:
        logger.error("Unsupported format: %s (expected one of %s)", suffix, ", ".join(sorted(IMAGE_FORMATS)))
        sys.exit(1)

    base_output = Path(args.output_dir).resolve()
    out_dir = unique_output_dir(base_output, file_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to PNG if needed (for local CLI or non-native formats)
    if suffix not in MINERU_NATIVE_FORMATS and not args.remote:
        logger.info("Format %s not natively supported by MinerU; converting to PNG ...", suffix)
        input_path = convert_to_png(str(file_path), str(out_dir))
    else:
        input_path = file_path

    # Step 2: Run MinerU
    mineru_output = out_dir / "mineru"
    if args.remote:
        run_mineru_remote(
            str(input_path), str(mineru_output),
            parse_method="auto", lang=args.lang or "en",
            backend=args.backend or "hybrid-auto-engine",
        )
    else:
        run_mineru(str(input_path), str(mineru_output), mode="ocr",
                   lang=args.lang, backend=args.backend, device=args.device)

    # Step 3: Read MinerU output
    content_list = read_mineru_output(str(mineru_output), input_path.stem)
    logger.info("Loaded %d content items.", len(content_list))

    # Step 4: Separate text vs multimodal
    full_text, multimodal_items = separate_content(content_list)

    # Step 5: Build and output chunks
    chunks = build_chunks(full_text, multimodal_items, str(file_path),
                          chunk_size=args.chunk_size, use_llm=args.use_llm)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print_summary(chunks, file_path, out_dir)


if __name__ == "__main__":
    main()
