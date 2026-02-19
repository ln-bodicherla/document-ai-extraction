"""
OCR pipeline for document text extraction and preprocessing.

Provides configurable Tesseract OCR execution, confidence-based filtering,
token correction, table detection, and spatial line merging.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRToken:
    """Single OCR-detected word with spatial and confidence metadata."""

    text: str
    confidence: int
    bbox: list[int]
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    page_num: int = 1


@dataclass
class OCRLine:
    """A group of OCR tokens forming a text line."""

    text: str
    tokens: list[OCRToken]
    bbox: list[int]
    avg_confidence: float


@dataclass
class TableCell:
    """A single cell within a detected table."""

    row: int
    col: int
    text: str
    bbox: list[int]


@dataclass
class DetectedTable:
    """A table structure detected from OCR bounding boxes."""

    cells: list[TableCell]
    num_rows: int
    num_cols: int
    bbox: list[int]


class OCRPipeline:
    """Configurable OCR pipeline for extracting and structuring document text.

    Wraps Tesseract OCR with preprocessing, confidence filtering, token
    correction, line merging, and heuristic table detection.
    """

    COMMON_CORRECTIONS = {
        "lnvoice": "Invoice",
        "Arnount": "Amount",
        "Totol": "Total",
        "Dote": "Date",
        "Qty": "Qty",
        "O0": "00",
        "l0": "10",
        "0l": "01",
        "|": "I",
    }

    NUMBER_PATTERN = re.compile(r"^[\$\€\£]?\d[\d,]*\.?\d*$")
    DATE_PATTERN = re.compile(
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|"
        r"\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|"
        r"\w{3,9}\s+\d{1,2},?\s+\d{4}"
    )

    def __init__(
        self,
        tesseract_config: str = "--psm 6 --oem 3",
        confidence_threshold: int = 60,
        language: str = "eng",
    ):
        self.tesseract_config = tesseract_config
        self.confidence_threshold = confidence_threshold
        self.language = language

    def run_tesseract(
        self,
        image: Image.Image,
        psm_mode: Optional[int] = None,
        language: Optional[str] = None,
    ) -> list[OCRToken]:
        """Execute Tesseract OCR on an image and return word-level tokens.

        Args:
            image: PIL Image to process.
            psm_mode: Override Page Segmentation Mode (1-13).
            language: Override language code.

        Returns:
            List of OCRToken objects for each detected word.
        """
        config = self.tesseract_config
        if psm_mode is not None:
            config = re.sub(r"--psm\s+\d+", f"--psm {psm_mode}", config)

        lang = language or self.language

        if image.mode != "RGB":
            image = image.convert("RGB")

        ocr_data = pytesseract.image_to_data(
            image,
            config=config,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        tokens = []
        num_entries = len(ocr_data["text"])

        for i in range(num_entries):
            text = ocr_data["text"][i].strip()
            conf = int(ocr_data["conf"][i])

            if not text or conf < 0:
                continue

            token = OCRToken(
                text=text,
                confidence=conf,
                bbox=[
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["left"][i] + ocr_data["width"][i],
                    ocr_data["top"][i] + ocr_data["height"][i],
                ],
                block_num=ocr_data["block_num"][i],
                par_num=ocr_data["par_num"][i],
                line_num=ocr_data["line_num"][i],
                word_num=ocr_data["word_num"][i],
                page_num=ocr_data.get("page_num", [1] * num_entries)[i],
            )
            tokens.append(token)

        logger.info("Tesseract returned %d tokens", len(tokens))
        return tokens

    def confidence_filter(
        self,
        tokens: list[OCRToken],
        threshold: Optional[int] = None,
    ) -> list[OCRToken]:
        """Filter OCR tokens below a confidence threshold.

        Args:
            tokens: List of OCR tokens.
            threshold: Minimum confidence (0-100). Uses instance default if None.

        Returns:
            Filtered list of tokens meeting the confidence requirement.
        """
        min_conf = threshold if threshold is not None else self.confidence_threshold
        filtered = [t for t in tokens if t.confidence >= min_conf]
        removed = len(tokens) - len(filtered)
        if removed > 0:
            logger.debug(
                "Confidence filter removed %d / %d tokens (threshold=%d)",
                removed, len(tokens), min_conf,
            )
        return filtered

    def correct_tokens(self, tokens: list[OCRToken]) -> list[OCRToken]:
        """Apply basic corrections to OCR token text.

        Performs common OCR error corrections and normalizes number formatting.

        Args:
            tokens: List of OCR tokens.

        Returns:
            Tokens with corrected text values.
        """
        corrected = []
        for token in tokens:
            text = token.text

            if text in self.COMMON_CORRECTIONS:
                text = self.COMMON_CORRECTIONS[text]

            text = self._normalize_number(text)
            text = self._fix_common_ocr_errors(text)

            corrected.append(
                OCRToken(
                    text=text,
                    confidence=token.confidence,
                    bbox=token.bbox,
                    block_num=token.block_num,
                    par_num=token.par_num,
                    line_num=token.line_num,
                    word_num=token.word_num,
                    page_num=token.page_num,
                )
            )
        return corrected

    def _normalize_number(self, text: str) -> str:
        """Normalize common OCR number errors."""
        cleaned = text.replace("O", "0").replace("l", "1").replace("I", "1")
        if self.NUMBER_PATTERN.match(cleaned) and not self.NUMBER_PATTERN.match(text):
            if any(c.isdigit() for c in text):
                return cleaned
        return text

    @staticmethod
    def _fix_common_ocr_errors(text: str) -> str:
        """Fix character-level OCR misrecognitions."""
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def extract_tables(
        self,
        tokens: list[OCRToken],
        row_tolerance: int = 15,
        col_tolerance: int = 30,
    ) -> list[DetectedTable]:
        """Detect table structures from OCR token bounding boxes.

        Uses spatial heuristics to group tokens into rows and columns,
        identifying tabular regions within the document.

        Args:
            tokens: List of OCR tokens with bounding boxes.
            row_tolerance: Vertical pixel tolerance for row grouping.
            col_tolerance: Horizontal pixel tolerance for column alignment.

        Returns:
            List of detected tables with cell contents.
        """
        if not tokens:
            return []

        rows = self._group_into_rows(tokens, row_tolerance)

        if len(rows) < 2:
            return []

        col_positions = self._detect_column_positions(rows, col_tolerance)

        if len(col_positions) < 2:
            return []

        tables = self._build_tables(rows, col_positions, col_tolerance)

        logger.info("Detected %d table(s)", len(tables))
        return tables

    def _group_into_rows(
        self, tokens: list[OCRToken], tolerance: int
    ) -> list[list[OCRToken]]:
        """Group tokens into horizontal rows based on vertical position."""
        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
        rows: list[list[OCRToken]] = []
        current_row: list[OCRToken] = []
        current_y = -1

        for token in sorted_tokens:
            y_center = (token.bbox[1] + token.bbox[3]) // 2

            if current_y < 0 or abs(y_center - current_y) <= tolerance:
                current_row.append(token)
                if current_y < 0:
                    current_y = y_center
                else:
                    current_y = (current_y + y_center) // 2
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda t: t.bbox[0]))
                current_row = [token]
                current_y = y_center

        if current_row:
            rows.append(sorted(current_row, key=lambda t: t.bbox[0]))

        return rows

    def _detect_column_positions(
        self, rows: list[list[OCRToken]], tolerance: int
    ) -> list[int]:
        """Identify column x-positions from token alignment across rows."""
        x_starts: list[int] = []
        for row in rows:
            for token in row:
                x_starts.append(token.bbox[0])

        x_starts.sort()
        columns: list[int] = []

        for x in x_starts:
            if not columns or abs(x - columns[-1]) > tolerance:
                columns.append(x)

        col_counts = {col: 0 for col in columns}
        for row in rows:
            matched_cols = set()
            for token in row:
                for col in columns:
                    if abs(token.bbox[0] - col) <= tolerance:
                        matched_cols.add(col)
                        break
            for col in matched_cols:
                col_counts[col] += 1

        min_row_count = max(2, len(rows) // 3)
        valid_columns = [
            col for col, count in col_counts.items() if count >= min_row_count
        ]

        return sorted(valid_columns)

    def _build_tables(
        self,
        rows: list[list[OCRToken]],
        col_positions: list[int],
        col_tolerance: int,
    ) -> list[DetectedTable]:
        """Build table structures by mapping tokens to row/column cells."""
        cells: list[TableCell] = []
        table_tokens: list[OCRToken] = []

        for row_idx, row in enumerate(rows):
            for token in row:
                col_idx = self._find_column(token.bbox[0], col_positions, col_tolerance)
                if col_idx >= 0:
                    cells.append(TableCell(
                        row=row_idx,
                        col=col_idx,
                        text=token.text,
                        bbox=token.bbox,
                    ))
                    table_tokens.append(token)

        if not cells:
            return []

        all_boxes = [t.bbox for t in table_tokens]
        table_bbox = [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]

        num_rows = max(c.row for c in cells) + 1
        num_cols = max(c.col for c in cells) + 1

        return [DetectedTable(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            bbox=table_bbox,
        )]

    @staticmethod
    def _find_column(x: int, col_positions: list[int], tolerance: int) -> int:
        """Find the closest column index for a given x-coordinate."""
        for idx, col_x in enumerate(col_positions):
            if abs(x - col_x) <= tolerance:
                return idx
        return -1

    def merge_lines(
        self,
        tokens: list[OCRToken],
        vertical_tolerance: int = 10,
        horizontal_gap_max: int = 50,
    ) -> list[OCRLine]:
        """Merge word-level OCR tokens into line-level text with spatial grouping.

        Groups tokens that share the same vertical position (within tolerance)
        and have reasonable horizontal spacing into coherent text lines.

        Args:
            tokens: List of OCR tokens.
            vertical_tolerance: Max vertical pixel distance for same-line grouping.
            horizontal_gap_max: Max horizontal gap between consecutive words.

        Returns:
            List of OCRLine objects representing merged text lines.
        """
        if not tokens:
            return []

        rows = self._group_into_rows(tokens, vertical_tolerance)
        lines: list[OCRLine] = []

        for row_tokens in rows:
            segments = self._split_row_by_gap(row_tokens, horizontal_gap_max)

            for segment in segments:
                text = " ".join(t.text for t in segment)
                avg_conf = sum(t.confidence for t in segment) / len(segment)

                merged_bbox = [
                    min(t.bbox[0] for t in segment),
                    min(t.bbox[1] for t in segment),
                    max(t.bbox[2] for t in segment),
                    max(t.bbox[3] for t in segment),
                ]

                lines.append(OCRLine(
                    text=text,
                    tokens=segment,
                    bbox=merged_bbox,
                    avg_confidence=avg_conf,
                ))

        return lines

    @staticmethod
    def _split_row_by_gap(
        tokens: list[OCRToken], max_gap: int
    ) -> list[list[OCRToken]]:
        """Split a row of tokens into segments based on horizontal gaps."""
        if not tokens:
            return []

        segments: list[list[OCRToken]] = [[tokens[0]]]

        for i in range(1, len(tokens)):
            prev_right = tokens[i - 1].bbox[2]
            curr_left = tokens[i].bbox[0]
            gap = curr_left - prev_right

            if gap > max_gap:
                segments.append([tokens[i]])
            else:
                segments[-1].append(tokens[i])

        return segments

    def full_pipeline(
        self, image: Image.Image
    ) -> dict[str, Any]:
        """Run the complete OCR pipeline on an image.

        Executes OCR, filters by confidence, corrects tokens, merges lines,
        and attempts table detection.

        Returns:
            Dictionary with 'tokens', 'lines', 'tables', and 'full_text' keys.
        """
        raw_tokens = self.run_tesseract(image)
        filtered_tokens = self.confidence_filter(raw_tokens)
        corrected_tokens = self.correct_tokens(filtered_tokens)
        lines = self.merge_lines(corrected_tokens)
        tables = self.extract_tables(corrected_tokens)

        full_text = "\n".join(line.text for line in lines)

        return {
            "tokens": corrected_tokens,
            "lines": lines,
            "tables": tables,
            "full_text": full_text,
            "token_count": len(corrected_tokens),
            "line_count": len(lines),
            "table_count": len(tables),
        }
