"""
LayoutLMv3-based document entity extraction.

Uses LayoutLMv3ForTokenClassification to extract structured fields from
document images by jointly reasoning over text, layout, and visual features.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents a single extracted entity from a document."""

    label: str
    value: str
    confidence: float
    bbox: list[int]
    token_indices: list[int] = field(default_factory=list)


class LayoutLMExtractor:
    """Extracts structured entities from document images using LayoutLMv3.

    Combines Tesseract OCR output (words + bounding boxes) with LayoutLMv3
    token classification to identify and extract key document fields such as
    vendor names, amounts, dates, and invoice numbers.
    """

    LABEL_LIST = [
        "O",
        "B-VENDOR", "I-VENDOR",
        "B-AMOUNT", "I-AMOUNT",
        "B-DATE", "I-DATE",
        "B-INVOICE_NUM", "I-INVOICE_NUM",
        "B-LINE_ITEM", "I-LINE_ITEM",
        "B-TAX", "I-TAX",
        "B-TOTAL", "I-TOTAL",
    ]

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
        label_list: Optional[list[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list or self.LABEL_LIST
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self) -> None:
        """Load the LayoutLMv3 model and processor."""
        logger.info("Loading LayoutLMv3 model: %s", self.model_name)
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name, apply_ocr=False
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded on device: %s", self.device)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    def preprocess_document(
        self,
        image: Image.Image,
        tesseract_config: str = "--psm 6 --oem 3",
        language: str = "eng",
    ) -> dict[str, Any]:
        """Run OCR on the image and prepare inputs for LayoutLMv3.

        Args:
            image: PIL Image of the document page.
            tesseract_config: Tesseract configuration string.
            language: OCR language code.

        Returns:
            Dictionary with 'words', 'boxes', and 'ocr_data' keys.
        """
        ocr_data = pytesseract.image_to_data(
            image,
            config=tesseract_config,
            lang=language,
            output_type=pytesseract.Output.DICT,
        )

        words = []
        boxes = []
        confidences = []
        width, height = image.size

        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            conf = int(ocr_data["conf"][i])
            if not text or conf < 0:
                continue

            left = ocr_data["left"][i]
            top = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            normalized_box = self.normalize_bbox(
                [left, top, left + w, top + h], width, height
            )
            words.append(text)
            boxes.append(normalized_box)
            confidences.append(conf)

        return {
            "words": words,
            "boxes": boxes,
            "confidences": confidences,
            "image": image,
            "original_size": (width, height),
        }

    @staticmethod
    def normalize_bbox(
        bbox: list[int], width: int, height: int
    ) -> list[int]:
        """Normalize bounding box coordinates to the 0-1000 range.

        Args:
            bbox: [x0, y0, x1, y1] in pixel coordinates.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Normalized [x0, y0, x1, y1] in the 0-1000 range.
        """
        if width == 0 or height == 0:
            return [0, 0, 0, 0]
        return [
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height),
        ]

    def _prepare_model_inputs(
        self, preprocessed: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Convert preprocessed OCR output into model-ready tensors."""
        encoding = self.processor(
            images=preprocessed["image"],
            text=preprocessed["words"],
            boxes=preprocessed["boxes"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def extract_entities(
        self,
        image: Image.Image,
        tesseract_config: str = "--psm 6 --oem 3",
        language: str = "eng",
        confidence_threshold: float = 0.5,
    ) -> list[ExtractedEntity]:
        """Extract structured entities from a document image.

        Args:
            image: PIL Image of the document page.
            tesseract_config: Tesseract configuration flags.
            language: OCR language code.
            confidence_threshold: Minimum confidence to include an entity.

        Returns:
            List of ExtractedEntity objects.
        """
        preprocessed = self.preprocess_document(image, tesseract_config, language)
        if not preprocessed["words"]:
            logger.warning("No words detected by OCR")
            return []

        inputs = self._prepare_model_inputs(preprocessed)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        max_probs = probabilities.max(dim=-1).values.squeeze().cpu().numpy()

        entities = self._group_bio_tokens(
            predictions,
            max_probs,
            preprocessed["words"],
            preprocessed["boxes"],
        )

        filtered = [e for e in entities if e.confidence >= confidence_threshold]
        merged = self._merge_subword_tokens(filtered)

        logger.info("Extracted %d entities from document", len(merged))
        return merged

    def _group_bio_tokens(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        words: list[str],
        boxes: list[list[int]],
    ) -> list[ExtractedEntity]:
        """Group B-I-O tagged tokens into complete entities.

        Iterates through token predictions, starting a new entity on each B-
        tag and extending it on matching I- tags.
        """
        entities: list[ExtractedEntity] = []
        current_entity: Optional[dict] = None
        word_idx = 0

        for token_idx in range(len(predictions)):
            if word_idx >= len(words):
                break

            label_id = int(predictions[token_idx])
            prob = float(probabilities[token_idx])
            label = self.id2label.get(label_id, "O")

            if label.startswith("B-"):
                if current_entity is not None:
                    entities.append(self._build_entity(current_entity))
                entity_type = label[2:]
                current_entity = {
                    "label": entity_type,
                    "tokens": [words[word_idx]],
                    "probs": [prob],
                    "boxes": [boxes[word_idx]],
                    "indices": [word_idx],
                }
                word_idx += 1

            elif label.startswith("I-") and current_entity is not None:
                expected_type = label[2:]
                if expected_type == current_entity["label"]:
                    current_entity["tokens"].append(words[word_idx])
                    current_entity["probs"].append(prob)
                    current_entity["boxes"].append(boxes[word_idx])
                    current_entity["indices"].append(word_idx)
                    word_idx += 1
                else:
                    entities.append(self._build_entity(current_entity))
                    current_entity = None
                    word_idx += 1
            else:
                if current_entity is not None:
                    entities.append(self._build_entity(current_entity))
                    current_entity = None
                if label == "O":
                    word_idx += 1

        if current_entity is not None:
            entities.append(self._build_entity(current_entity))

        return entities

    @staticmethod
    def _build_entity(entity_data: dict) -> ExtractedEntity:
        """Construct an ExtractedEntity from grouped token data."""
        value = " ".join(entity_data["tokens"])
        confidence = float(np.mean(entity_data["probs"]))

        all_boxes = entity_data["boxes"]
        merged_box = [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]

        return ExtractedEntity(
            label=entity_data["label"],
            value=value,
            confidence=confidence,
            bbox=merged_box,
            token_indices=entity_data["indices"],
        )

    def _merge_subword_tokens(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Merge sub-word token artifacts in entity values.

        Cleans up tokenization artifacts like '##' prefixes and re-joins
        hyphenated fragments produced by the tokenizer.
        """
        merged = []
        for entity in entities:
            cleaned_value = entity.value.replace(" ##", "")
            cleaned_value = cleaned_value.replace("##", "")
            cleaned_value = cleaned_value.replace(" - ", "-")
            cleaned_value = cleaned_value.strip()

            if not cleaned_value:
                continue

            merged.append(
                ExtractedEntity(
                    label=entity.label,
                    value=cleaned_value,
                    confidence=entity.confidence,
                    bbox=entity.bbox,
                    token_indices=entity.token_indices,
                )
            )
        return merged

    def extract_to_dict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Extract entities and return them as plain dictionaries.

        Convenience method for serialization and API responses.
        """
        entities = self.extract_entities(
            image, confidence_threshold=confidence_threshold
        )
        return [
            {
                "label": e.label,
                "value": e.value,
                "confidence": round(e.confidence, 4),
                "bbox": e.bbox,
            }
            for e in entities
        ]

    def batch_extract(
        self,
        images: list[Image.Image],
        confidence_threshold: float = 0.5,
    ) -> list[list[dict[str, Any]]]:
        """Extract entities from multiple document images.

        Processes each image sequentially. For GPU-bound workloads,
        batching is handled at the API layer.
        """
        results = []
        for idx, image in enumerate(images):
            logger.info("Processing image %d / %d", idx + 1, len(images))
            entities = self.extract_to_dict(image, confidence_threshold)
            results.append(entities)
        return results
