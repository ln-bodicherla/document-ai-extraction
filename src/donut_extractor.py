"""
Donut-based document understanding and extraction.

Uses the Donut (Document Understanding Transformer) model to generate
structured JSON output directly from document images without requiring
an external OCR engine.
"""

import json
import logging
import re
from typing import Any, Optional

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


class DonutExtractor:
    """Extracts structured data from document images using the Donut model.

    Donut is an end-to-end document understanding transformer that combines
    a Swin Transformer visual encoder with a BART text decoder. It generates
    structured JSON output directly from document images, bypassing the need
    for a separate OCR step.
    """

    DEFAULT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"

    TASK_PROMPTS = {
        "cord": "<s_cord-v2>",
        "invoice": "<s_invoice>",
        "receipt": "<s_receipt>",
        "form": "<s_form>",
        "general": "<s_docvqa>",
    }

    FIELD_MAPPING = {
        "menu.nm": "LINE_ITEM",
        "menu.unitprice": "AMOUNT",
        "menu.cnt": "QUANTITY",
        "menu.price": "AMOUNT",
        "menu.sub_nm": "LINE_ITEM",
        "menu.sub_unitprice": "AMOUNT",
        "menu.sub_cnt": "QUANTITY",
        "menu.sub_price": "AMOUNT",
        "menu.etc": "LINE_ITEM",
        "menu.vatyn": "TAX",
        "total.total_price": "TOTAL",
        "total.total_etc": "TOTAL",
        "total.cashprice": "AMOUNT",
        "total.changeprice": "AMOUNT",
        "total.creditcardprice": "AMOUNT",
        "total.emoneyprice": "AMOUNT",
        "total.menutype_cnt": "QUANTITY",
        "total.menuqty_cnt": "QUANTITY",
        "sub_total.subtotal_price": "AMOUNT",
        "sub_total.tax_price": "TAX",
        "sub_total.discount_price": "AMOUNT",
        "sub_total.service_price": "AMOUNT",
        "sub_total.etc": "AMOUNT",
        "void_menu.nm": "LINE_ITEM",
        "void_menu.price": "AMOUNT",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 768,
        task_prompt: str = "<s_cord-v2>",
        device: Optional[str] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.task_prompt = task_prompt

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self) -> None:
        """Load the Donut model and processor from HuggingFace."""
        logger.info("Loading Donut model: %s", self.model_name)
        try:
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Donut model loaded on device: %s", self.device)
        except Exception as exc:
            logger.error("Failed to load Donut model: %s", exc)
            raise

    def preprocess_image(
        self,
        image: Image.Image,
        target_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """Preprocess a document image for Donut input.

        Resizes and normalizes the image according to the processor's
        expected input format. Donut expects fixed-size image inputs.

        Args:
            image: PIL Image of the document.
            target_size: Optional (width, height) override for resizing.

        Returns:
            Dictionary of tensors ready for model input.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        if target_size is not None:
            image = image.resize(target_size, Image.LANCZOS)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        return {"pixel_values": pixel_values}

    def extract_structured(
        self,
        image: Image.Image,
        task_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output from a document image.

        Runs the Donut model to produce a structured representation of the
        document's content, parsed from the generated token sequence.

        Args:
            image: PIL Image of the document.
            task_prompt: Override for the task-specific start token.

        Returns:
            Dictionary containing the parsed structured output.
        """
        prompt = task_prompt or self.task_prompt
        inputs = self.preprocess_image(image)

        decoder_input_ids = self.processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["pixel_values"],
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_sequence = outputs.sequences[0]
        decoded_text = self.processor.tokenizer.decode(
            generated_sequence, skip_special_tokens=False
        )

        parsed = self.parse_output(decoded_text, prompt)
        confidence = self._compute_sequence_confidence(outputs)

        return {
            "raw_output": decoded_text,
            "parsed": parsed,
            "confidence": confidence,
        }

    def parse_output(
        self, generated_text: str, task_prompt: str
    ) -> dict[str, Any]:
        """Parse Donut's generated token sequence into structured fields.

        The model generates XML-like tagged output that is converted into
        a nested dictionary structure.

        Args:
            generated_text: Raw decoded text from the model.
            task_prompt: The task prompt used (stripped from output).

        Returns:
            Nested dictionary of extracted fields.
        """
        cleaned = generated_text.replace(self.processor.tokenizer.eos_token, "")
        cleaned = cleaned.replace(self.processor.tokenizer.pad_token, "")
        cleaned = re.sub(r"<s_\w+>", "", cleaned)
        cleaned = cleaned.strip()

        return self._parse_xml_tokens(cleaned)

    def _parse_xml_tokens(self, text: str) -> dict[str, Any]:
        """Recursively parse XML-like token output into a dictionary.

        Handles nested structures like <s_menu><s_nm>item</s_nm></s_menu>.
        """
        result = {}
        pattern = r"<s_(\w+)>(.*?)</s_\1>"
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            key = match.group(1)
            value_text = match.group(2).strip()

            if "<s_" in value_text:
                nested = self._parse_xml_tokens(value_text)
                if key in result:
                    if isinstance(result[key], list):
                        result[key].append(nested)
                    else:
                        result[key] = [result[key], nested]
                else:
                    result[key] = nested
            else:
                if "<sep/>" in value_text:
                    values = [v.strip() for v in value_text.split("<sep/>") if v.strip()]
                    result[key] = values
                else:
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value_text)
                        else:
                            result[key] = [result[key], value_text]
                    else:
                        result[key] = value_text

        return result

    def _compute_sequence_confidence(self, outputs: Any) -> float:
        """Compute an aggregate confidence score for the generated sequence.

        Uses the mean of token-level log-probabilities converted to a
        probability score.
        """
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.0

        log_probs = []
        for step_scores in outputs.scores:
            probs = torch.softmax(step_scores, dim=-1)
            max_prob = probs.max(dim=-1).values
            log_probs.append(max_prob.item())

        if not log_probs:
            return 0.0

        return float(sum(log_probs) / len(log_probs))

    def flatten_parsed_output(
        self, parsed: dict[str, Any], prefix: str = ""
    ) -> list[dict[str, Any]]:
        """Flatten nested parsed output into a list of field-value pairs.

        Converts hierarchical Donut output into a flat list suitable for
        comparison with LayoutLM entity extraction results.

        Args:
            parsed: Nested dictionary from parse_output().
            prefix: Key prefix for nested fields.

        Returns:
            List of dicts with 'field', 'value', and 'mapped_label' keys.
        """
        fields = []

        for key, value in parsed.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                fields.extend(self.flatten_parsed_output(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        fields.extend(
                            self.flatten_parsed_output(item, full_key)
                        )
                    else:
                        mapped_label = self.FIELD_MAPPING.get(full_key, "OTHER")
                        fields.append({
                            "field": full_key,
                            "value": str(item),
                            "mapped_label": mapped_label,
                            "index": i,
                        })
            else:
                mapped_label = self.FIELD_MAPPING.get(full_key, "OTHER")
                fields.append({
                    "field": full_key,
                    "value": str(value),
                    "mapped_label": mapped_label,
                })

        return fields

    def extract_to_entities(
        self,
        image: Image.Image,
        task_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Extract and convert Donut output into standardized entity format.

        Produces output compatible with the LayoutLM extractor's entity format
        for ensemble merging.

        Args:
            image: PIL Image of the document.
            task_prompt: Optional task prompt override.

        Returns:
            List of entity dictionaries with label, value, and confidence.
        """
        result = self.extract_structured(image, task_prompt)
        flat_fields = self.flatten_parsed_output(result["parsed"])
        base_confidence = result["confidence"]

        entities = []
        for field_data in flat_fields:
            entities.append({
                "label": field_data["mapped_label"],
                "value": field_data["value"],
                "confidence": round(base_confidence, 4),
                "bbox": [0, 0, 0, 0],
                "source": "donut",
                "original_field": field_data["field"],
            })

        return entities

    def batch_extract(
        self,
        images: list[Image.Image],
        task_prompt: Optional[str] = None,
    ) -> list[list[dict[str, Any]]]:
        """Process multiple document images sequentially.

        Args:
            images: List of PIL Images.
            task_prompt: Optional task prompt override.

        Returns:
            List of entity lists, one per input image.
        """
        all_results = []
        for idx, image in enumerate(images):
            logger.info("Donut processing image %d / %d", idx + 1, len(images))
            entities = self.extract_to_entities(image, task_prompt)
            all_results.append(entities)
        return all_results

    def set_task(self, task_name: str) -> None:
        """Switch the extraction task by name.

        Args:
            task_name: One of 'cord', 'invoice', 'receipt', 'form', 'general'.

        Raises:
            ValueError: If the task name is not recognized.
        """
        if task_name not in self.TASK_PROMPTS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {list(self.TASK_PROMPTS.keys())}"
            )
        self.task_prompt = self.TASK_PROMPTS[task_name]
        logger.info("Switched to task '%s' with prompt '%s'", task_name, self.task_prompt)
