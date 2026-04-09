#!/usr/bin/env python3
"""Parse image files via MinerU OCR and produce chunk JSON.

Supports: .png, .jpg, .jpeg, .bmp, .tiff, .tif, .gif, .webp
"""

import logging
from pathlib import Path

from base_parser import BaseParser, ParserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
MINERU_NATIVE_FORMATS = {".png", ".jpeg", ".jpg"}


class ImageParser(BaseParser):
    """Parse image files (.png/.jpg/.jpeg/.bmp/.tiff/.tif/.gif/.webp) via MinerU OCR."""

    def supported_formats(self) -> set[str]:
        return IMAGE_FORMATS

    def _is_image_parser(self) -> bool:
        return True

    def preprocess(self, file_path: Path, out_dir: Path) -> tuple[Path, list[dict]]:
        suffix = file_path.suffix.lower()

        # Remote API handles all formats; local CLI needs PNG for non-native
        if suffix not in MINERU_NATIVE_FORMATS and not self.config.remote:
            logger.info("Converting %s to PNG for MinerU compatibility ...", suffix)
            return self._convert_to_png(file_path, out_dir), []

        return file_path, []

    @staticmethod
    def _convert_to_png(image_path: Path, output_dir: Path) -> Path:
        from PIL import Image

        img = Image.open(image_path)

        if img.mode in ("RGBA", "LA", "P"):
            if img.mode == "P":
                img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        png_path = output_dir / f"{image_path.stem}.png"
        img.save(str(png_path), format="PNG")
        logger.info("Converted to %s", png_path)
        return png_path


if __name__ == "__main__":
    ImageParser(ParserConfig()).run_cli()
