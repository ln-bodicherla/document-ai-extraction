"""
Document processing orchestrator.

Coordinates OCR, LayoutLMv3, and Donut extraction pipelines to produce
unified structured output from various document formats.
"""

import io
import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class DocumentProcessor:
    """Orchestrates document ingestion and field extraction.

    Supports PDF, PNG, JPG, and TIFF inputs. Routes documents through
    LayoutLMv3 and/or Donut extraction backends. An ensemble mode runs
    both models and merges results by per-field confidence.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        extraction_mode: str = "ensemble",
    ):
        """Initialize the document processor.

        Args:
            config_path: Path to YAML configuration file.
            extraction_mode: One of 'layoutlm', 'donut', or 'ensemble'.
        """
        self.config = self._load_config(config_path)
        self.extraction_mode = extraction_mode
        self._layoutlm_extractor = None
        self._donut_extractor = None
        self._ocr_pipeline = None

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict[str, Any]:
        """Load configuration from a YAML file."""
        if config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file not found: %s, using defaults", config_path)
            return {}
        with open(path) as f:
            config = yaml.safe_load(f)
        return config or {}

    @property
    def layoutlm_extractor(self):
        """Lazy-load the LayoutLM extractor."""
        if self._layoutlm_extractor is None:
            from src.layoutlm_extractor import LayoutLMExtractor

            lm_config = self.config.get("layoutlm", {})
            self._layoutlm_extractor = LayoutLMExtractor(
                model_name=lm_config.get("model_name", "microsoft/layoutlmv3-base"),
                max_seq_length=lm_config.get("max_seq_length", 512),
                label_list=lm_config.get("label_list"),
            )
        return self._layoutlm_extractor

    @property
    def donut_extractor(self):
        """Lazy-load the Donut extractor."""
        if self._donut_extractor is None:
            from src.donut_extractor import DonutExtractor

            donut_config = self.config.get("donut", {})
            self._donut_extractor = DonutExtractor(
                model_name=donut_config.get(
                    "model_name", "naver-clova-ix/donut-base-finetuned-cord-v2"
                ),
                max_length=donut_config.get("max_length", 768),
                task_prompt=donut_config.get("task_prompt", "<s_cord-v2>"),
            )
        return self._donut_extractor

    @property
    def ocr_pipeline(self):
        """Lazy-load the OCR pipeline."""
        if self._ocr_pipeline is None:
            from src.ocr_pipeline import OCRPipeline

            ocr_config = self.config.get("ocr", {})
            self._ocr_pipeline = OCRPipeline(
                tesseract_config=ocr_config.get(
                    "tesseract_config", "--psm 6 --oem 3"
                ),
                confidence_threshold=ocr_config.get("confidence_threshold", 60),
                language=ocr_config.get("language", "eng"),
            )
        return self._ocr_pipeline

    def process_document(
        self,
        source: Union[str, bytes, Path, Image.Image],
        extraction_mode: Optional[str] = None,
        page_numbers: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Process a document and extract structured fields.

        This is the main entry point. Accepts file paths, raw bytes,
        or PIL Images. PDFs are split into pages and each page is
        processed independently.

        Args:
            source: File path, bytes content, or PIL Image.
            extraction_mode: Override instance extraction mode.
            page_numbers: Specific pages to process (1-indexed, PDF only).

        Returns:
            Dictionary with 'entities', 'pages', 'metadata' keys.
        """
        mode = extraction_mode or self.extraction_mode
        images = self._load_images(source, page_numbers)

        if not images:
            return {
                "entities": [],
                "pages": [],
                "metadata": {"error": "No images could be loaded from source"},
            }

        all_entities = []
        page_results = []

        for page_idx, image in enumerate(images):
            image = self._preprocess_image(image)
            logger.info(
                "Processing page %d / %d (%dx%d)",
                page_idx + 1, len(images), image.width, image.height,
            )

            page_entities = self._extract_page(image, mode)
            page_results.append({
                "page_number": page_idx + 1,
                "entities": page_entities,
                "image_size": [image.width, image.height],
            })
            all_entities.extend(page_entities)

        deduplicated = self._deduplicate_entities(all_entities)

        metadata = {
            "total_pages": len(images),
            "extraction_mode": mode,
            "total_entities": len(deduplicated),
        }

        return {
            "entities": deduplicated,
            "pages": page_results,
            "metadata": metadata,
        }

    def _extract_page(
        self, image: Image.Image, mode: str
    ) -> list[dict[str, Any]]:
        """Run extraction on a single page image using the specified mode."""
        if mode == "layoutlm":
            return self._run_layoutlm(image)
        elif mode == "donut":
            return self._run_donut(image)
        elif mode == "ensemble":
            return self._run_ensemble(image)
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")

    def _run_layoutlm(self, image: Image.Image) -> list[dict[str, Any]]:
        """Extract entities using LayoutLMv3."""
        try:
            entities = self.layoutlm_extractor.extract_to_dict(image)
            for entity in entities:
                entity["source"] = "layoutlm"
            return entities
        except Exception as exc:
            logger.error("LayoutLM extraction failed: %s", exc)
            return []

    def _run_donut(self, image: Image.Image) -> list[dict[str, Any]]:
        """Extract entities using Donut."""
        try:
            entities = self.donut_extractor.extract_to_entities(image)
            return entities
        except Exception as exc:
            logger.error("Donut extraction failed: %s", exc)
            return []

    def _run_ensemble(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run both extractors and merge results by confidence.

        For each entity type, keeps the extraction with the highest
        confidence score. Entities from different types are all included.
        """
        layoutlm_entities = self._run_layoutlm(image)
        donut_entities = self._run_donut(image)

        merged = self._merge_by_confidence(layoutlm_entities, donut_entities)
        return merged

    @staticmethod
    def _merge_by_confidence(
        entities_a: list[dict[str, Any]],
        entities_b: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge two entity lists, preferring higher-confidence extractions.

        Groups entities by label and selects the highest-confidence value
        for each label. For labels with multiple instances (e.g., LINE_ITEM),
        all instances are retained and sorted by confidence.
        """
        single_instance_labels = {
            "VENDOR", "DATE", "INVOICE_NUM", "TAX", "TOTAL"
        }
        multi_instance_labels = {"LINE_ITEM", "AMOUNT", "QUANTITY"}

        best_single: dict[str, dict[str, Any]] = {}
        multi_entities: list[dict[str, Any]] = []

        for entity in entities_a + entities_b:
            label = entity.get("label", "OTHER")

            if label in single_instance_labels:
                existing = best_single.get(label)
                if existing is None or entity["confidence"] > existing["confidence"]:
                    best_single[label] = entity
            elif label in multi_instance_labels:
                multi_entities.append(entity)
            else:
                multi_entities.append(entity)

        merged = list(best_single.values())
        merged.extend(multi_entities)

        merged.sort(key=lambda e: e.get("confidence", 0), reverse=True)
        return merged

    def _load_images(
        self,
        source: Union[str, bytes, Path, Image.Image],
        page_numbers: Optional[list[int]] = None,
    ) -> list[Image.Image]:
        """Load document source into a list of PIL Images.

        Handles file paths (including PDF), raw bytes, and direct Image input.
        """
        if isinstance(source, Image.Image):
            return [source]

        if isinstance(source, bytes):
            return self._load_from_bytes(source, page_numbers)

        path = Path(source) if isinstance(source, str) else source

        if not path.exists():
            logger.error("File not found: %s", path)
            return []

        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file type: %s", ext)
            return []

        if ext == ".pdf":
            return self._load_pdf(path, page_numbers)

        image = Image.open(path)
        return [image]

    def _load_pdf(
        self, path: Path, page_numbers: Optional[list[int]] = None
    ) -> list[Image.Image]:
        """Convert PDF pages to PIL Images using pdf2image."""
        try:
            from pdf2image import convert_from_path

            dpi = self.config.get("processing", {}).get("dpi", 300)

            kwargs = {"dpi": dpi}
            if page_numbers:
                kwargs["first_page"] = min(page_numbers)
                kwargs["last_page"] = max(page_numbers)

            images = convert_from_path(str(path), **kwargs)

            if page_numbers:
                offset = min(page_numbers) - 1
                selected = []
                for pn in page_numbers:
                    idx = pn - 1 - offset
                    if 0 <= idx < len(images):
                        selected.append(images[idx])
                return selected

            return images

        except ImportError:
            logger.error(
                "pdf2image is not installed. Install it with: pip install pdf2image"
            )
            return []
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", path, exc)
            return []

    def _load_from_bytes(
        self, data: bytes, page_numbers: Optional[list[int]] = None
    ) -> list[Image.Image]:
        """Load images from raw bytes (supports PDF and image formats)."""
        try:
            image = Image.open(io.BytesIO(data))
            return [image]
        except Exception:
            pass

        try:
            from pdf2image import convert_from_bytes

            dpi = self.config.get("processing", {}).get("dpi", 300)
            images = convert_from_bytes(data, dpi=dpi)

            if page_numbers:
                return [images[pn - 1] for pn in page_numbers if 0 < pn <= len(images)]
            return images
        except Exception as exc:
            logger.error("Failed to load from bytes: %s", exc)
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing transformations to the document image."""
        max_size = self.config.get("processing", {}).get("max_image_size", 2048)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug("Resized image to %s", new_size)

        return image

    @staticmethod
    def _deduplicate_entities(
        entities: list[dict[str, Any]],
        value_similarity_threshold: float = 0.9,
    ) -> list[dict[str, Any]]:
        """Remove duplicate entities with the same label and similar values."""
        if not entities:
            return []

        seen: dict[str, list[dict[str, Any]]] = {}

        for entity in entities:
            label = entity.get("label", "OTHER")
            value = entity.get("value", "")

            if label not in seen:
                seen[label] = [entity]
                continue

            is_duplicate = False
            for existing in seen[label]:
                existing_value = existing.get("value", "")
                if _text_similarity(value, existing_value) >= value_similarity_threshold:
                    if entity["confidence"] > existing["confidence"]:
                        seen[label].remove(existing)
                        seen[label].append(entity)
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen[label].append(entity)

        deduplicated = []
        for label_entities in seen.values():
            deduplicated.extend(label_entities)

        return sorted(deduplicated, key=lambda e: e.get("confidence", 0), reverse=True)

    def process_batch(
        self,
        sources: list[Union[str, bytes, Path]],
        extraction_mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Process multiple documents in sequence.

        Args:
            sources: List of file paths or byte contents.
            extraction_mode: Override extraction mode for all documents.

        Returns:
            List of extraction results, one per document.
        """
        results = []
        for idx, source in enumerate(sources):
            logger.info("Batch processing document %d / %d", idx + 1, len(sources))
            try:
                result = self.process_document(source, extraction_mode)
                result["metadata"]["batch_index"] = idx
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process document %d: %s", idx, exc)
                results.append({
                    "entities": [],
                    "pages": [],
                    "metadata": {
                        "batch_index": idx,
                        "error": str(exc),
                    },
                })
        return results


def _text_similarity(text_a: str, text_b: str) -> float:
    """Compute a simple character-level similarity ratio between two strings."""
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    a_lower = text_a.lower().strip()
    b_lower = text_b.lower().strip()

    if a_lower == b_lower:
        return 1.0

    shorter = min(len(a_lower), len(b_lower))
    longer = max(len(a_lower), len(b_lower))

    if longer == 0:
        return 1.0

    matches = sum(1 for ca, cb in zip(a_lower, b_lower) if ca == cb)
    return matches / longer
