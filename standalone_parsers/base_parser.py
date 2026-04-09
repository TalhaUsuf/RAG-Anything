"""
Base parser module implementing the Template Method + Strategy pattern.

Provides ``ParserConfig``, ``AIServices``, and the abstract ``BaseParser``
class.  Concrete parsers only need to implement ``supported_formats()`` and
``preprocess()``.  The rest of the pipeline (validation, MinerU execution,
content separation, chunk building, CLI wiring) is handled here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _common import (
    build_chunks,
    call_llm,
    call_vlm,
    convert_to_pdf_via_libreoffice,
    enhance_caption,
    print_summary,
    read_mineru_output,
    run_mineru,
    run_mineru_remote,
    separate_content,
    unique_output_dir,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParserConfig:
    """All tuneable knobs for a standalone parser run."""

    output_dir: str = "./output"
    remote: bool = False
    use_llm: bool = False
    chunk_size: int = 1200
    lang: str | None = None
    backend: str | None = None
    device: str | None = None


# ---------------------------------------------------------------------------
# AI service facade
# ---------------------------------------------------------------------------

class AIServices:
    """Encapsulates all external AI service calls.

    Delegates every call to the corresponding function in ``_common.py`` so
    that callers never import those helpers directly.
    """

    # -- MinerU -----------------------------------------------------------------

    @staticmethod
    def run_mineru(
        input_path: str,
        output_dir: str,
        *,
        mode: str = "auto",
        lang: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the local MinerU CLI."""
        run_mineru(
            input_path,
            output_dir,
            mode=mode,
            lang=lang,
            backend=backend,
            device=device,
            **kwargs,
        )

    @staticmethod
    def run_mineru_remote(
        input_path: str,
        output_dir: str,
        *,
        lang: str = "en",
        backend: str = "hybrid-auto-engine",
        **kwargs: Any,
    ) -> None:
        """Post a file to the remote MinerU HTTP API."""
        run_mineru_remote(
            input_path,
            output_dir,
            lang=lang,
            backend=backend,
            **kwargs,
        )

    # -- LLM / VLM --------------------------------------------------------------

    @staticmethod
    def call_llm(
        prompt: str,
        *,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> str:
        """Call an OpenAI-compatible text LLM."""
        return call_llm(prompt, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def call_vlm(
        prompt: str,
        image_base64: str,
        *,
        system_prompt: str = (
            "You are an expert image analyst. "
            "Provide detailed, accurate descriptions."
        ),
        **kwargs: Any,
    ) -> str:
        """Call a Vision LLM with a base64-encoded image."""
        return call_vlm(
            prompt,
            image_base64,
            system_prompt=system_prompt,
            **kwargs,
        )

    # -- High-level helpers -----------------------------------------------------

    @staticmethod
    def enhance_caption(
        item: dict,
        item_type: str,
        context: str = "",
    ) -> str:
        """Generate an LLM-enhanced caption for a multimodal item."""
        return enhance_caption(item, item_type, context=context)


# ---------------------------------------------------------------------------
# Abstract base parser (Template Method)
# ---------------------------------------------------------------------------

class BaseParser(ABC):
    """Base class for all standalone file parsers.

    Implements the **Template Method** pattern.  Subclasses only need to
    implement :meth:`supported_formats` and :meth:`preprocess`.  The rest
    of the pipeline is handled by :meth:`parse`.
    """

    def __init__(
        self,
        config: ParserConfig | None = None,
        services: AIServices | None = None,
    ) -> None:
        self.config = config or ParserConfig()
        self.services = services or AIServices()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def supported_formats(self) -> set[str]:
        """Return the set of supported file extensions (e.g. ``{'.png', '.jpg'}``)."""

    @abstractmethod
    def preprocess(
        self,
        file_path: Path,
        out_dir: Path,
    ) -> tuple[Path, list[dict]]:
        """Convert the input file to a format MinerU can process.

        Returns
        -------
        mineru_input_path : Path
            Path to the PDF or image file that should be sent to MinerU.
        extra_multimodal_items : list[dict]
            Any items extracted *before* MinerU (e.g. embedded images from
            a Markdown file) that should be merged into the final chunks.
        """

    # ------------------------------------------------------------------
    # Template method — the full parsing pipeline
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> list[dict]:
        """Run the complete parsing pipeline.

        Steps:
            1. Validate input file
            2. Set up output directory
            3. Preprocess (subclass hook)
            4. Run MinerU (local or remote)
            5. Read MinerU output
            6. Separate text / multimodal content
            7. Merge extra multimodal items from preprocessing
            8. Build chunks
            9. Return chunks
        """
        file_path_obj = Path(file_path).resolve()
        self._validate(file_path_obj)

        out_dir = unique_output_dir(
            Path(self.config.output_dir).resolve(),
            file_path_obj,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Preprocess (subclass hook)
        mineru_input, extra_multimodal = self.preprocess(file_path_obj, out_dir)

        # Step 4: Run MinerU
        mineru_output = out_dir / "mineru"
        if self.config.remote:
            self.services.run_mineru_remote(
                str(mineru_input),
                str(mineru_output),
                lang=self.config.lang or "en",
                backend=self.config.backend or "hybrid-auto-engine",
            )
        else:
            self.services.run_mineru(
                str(mineru_input),
                str(mineru_output),
                mode="ocr" if self._is_image_parser() else "auto",
                lang=self.config.lang,
                backend=self.config.backend,
                device=self.config.device,
            )

        # Step 5: Read output
        content_list = read_mineru_output(str(mineru_output), mineru_input.stem)

        # Step 6: Separate text from multimodal items
        full_text, mineru_multimodal = separate_content(content_list)

        # Step 7: Merge extra items from preprocessing
        all_multimodal = mineru_multimodal + extra_multimodal
        combined_content_list = content_list + extra_multimodal

        # Step 8: Build chunks
        chunks = build_chunks(
            full_text,
            all_multimodal,
            str(file_path_obj),
            chunk_size=self.config.chunk_size,
            use_llm=self.config.use_llm,
            content_list=combined_content_list,
        )

        return chunks

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, file_path: Path) -> None:
        """Check that the file exists and has a supported extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_formats():
            raise ValueError(
                f"Unsupported format: {file_path.suffix} "
                f"(expected one of {', '.join(sorted(self.supported_formats()))})"
            )

    # ------------------------------------------------------------------
    # Hooks for subclasses (optional overrides)
    # ------------------------------------------------------------------

    def _is_image_parser(self) -> bool:
        """Return ``True`` if MinerU should run in OCR mode.

        Override in image-based parsers.
        """
        return False

    def _add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Hook for subclasses to register extra CLI arguments."""

    def _apply_cli_args(self, args: argparse.Namespace) -> None:
        """Hook for subclasses to consume extra CLI arguments."""

    # ------------------------------------------------------------------
    # CLI entry-point
    # ------------------------------------------------------------------

    def run_cli(self) -> None:
        """Run the parser as a command-line tool with ``argparse``."""
        parser = argparse.ArgumentParser(
            description=(
                self.__class__.__doc__
                or "Parse files via the MinerU pipeline."
            ),
        )
        parser.add_argument("file", help="Path to the input file")
        parser.add_argument(
            "output_dir",
            nargs="?",
            default="./output",
            help="Output directory (default: ./output)",
        )
        parser.add_argument(
            "--remote",
            action="store_true",
            help="Use the remote MinerU API instead of the local CLI",
        )
        parser.add_argument(
            "--use-llm",
            action="store_true",
            help="Use LLM for enhanced multimodal captions",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=1200,
            help="Target chunk size in characters (default: 1200)",
        )
        parser.add_argument(
            "-l",
            "--lang",
            default=None,
            help="Language hint for MinerU",
        )
        parser.add_argument(
            "-b",
            "--backend",
            default=None,
            help="MinerU backend",
        )
        parser.add_argument(
            "-d",
            "--device",
            default=None,
            help="Device for MinerU (cpu/cuda)",
        )

        # Let subclasses add their own flags
        self._add_cli_args(parser)

        args = parser.parse_args()

        # Populate config from parsed arguments
        self.config.output_dir = args.output_dir
        self.config.remote = args.remote
        self.config.use_llm = args.use_llm
        self.config.chunk_size = args.chunk_size
        self.config.lang = args.lang
        self.config.backend = args.backend
        self.config.device = args.device

        # Let subclasses read their custom flags
        self._apply_cli_args(args)

        try:
            chunks = self.parse(args.file)
            print(json.dumps(chunks, indent=2, ensure_ascii=False))
            print_summary(chunks, args.file, self.config.output_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            sys.exit(1)
