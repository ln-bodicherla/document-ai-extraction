"""
FastAPI application for document AI extraction.

Provides REST endpoints for uploading documents and receiving
structured entity extraction results.
"""

import io
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
}

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExtractedEntityResponse(BaseModel):
    """Single extracted entity in the API response."""

    label: str = Field(..., description="Entity type (e.g., VENDOR, TOTAL)")
    value: str = Field(..., description="Extracted text value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Bounding box [x0, y0, x1, y1] in normalized coordinates",
    )
    source: Optional[str] = Field(
        None, description="Extraction backend (layoutlm, donut)"
    )


class PageResult(BaseModel):
    """Extraction results for a single document page."""

    page_number: int
    entities: list[ExtractedEntityResponse]
    image_size: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="[width, height] of the processed image",
    )


class ExtractionResponse(BaseModel):
    """Full extraction response for a single document."""

    request_id: str = Field(..., description="Unique request identifier")
    entities: list[ExtractedEntityResponse]
    pages: list[PageResult]
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )


class BatchExtractionResponse(BaseModel):
    """Response for batch document extraction."""

    request_id: str
    results: list[ExtractionResponse]
    total_documents: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document AI Extraction API",
    description=(
        "Multimodal document understanding pipeline for extracting structured "
        "fields from invoices, receipts, and forms using LayoutLMv3 and Donut."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_processor = None


def get_processor():
    """Lazily initialize the document processor singleton."""
    global _processor
    if _processor is None:
        from src.document_processor import DocumentProcessor

        config_path = "configs/config.yaml"
        if not Path(config_path).exists():
            config_path = None
        _processor = DocumentProcessor(config_path=config_path)
    return _processor


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> None:
    """Validate an uploaded file's type and extension."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: {file.content_type}. "
                    f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            )

    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file extension: {ext}",
            )


async def _read_upload(file: UploadFile) -> bytes:
    """Read upload contents with size validation."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return content


def _build_response(
    result: dict[str, Any], request_id: str, processing_time_ms: float
) -> ExtractionResponse:
    """Convert processor output to API response model."""
    entities = [
        ExtractedEntityResponse(
            label=e.get("label", "UNKNOWN"),
            value=e.get("value", ""),
            confidence=e.get("confidence", 0.0),
            bbox=e.get("bbox", [0, 0, 0, 0]),
            source=e.get("source"),
        )
        for e in result.get("entities", [])
    ]

    pages = []
    for page_data in result.get("pages", []):
        page_entities = [
            ExtractedEntityResponse(
                label=e.get("label", "UNKNOWN"),
                value=e.get("value", ""),
                confidence=e.get("confidence", 0.0),
                bbox=e.get("bbox", [0, 0, 0, 0]),
                source=e.get("source"),
            )
            for e in page_data.get("entities", [])
        ]
        pages.append(PageResult(
            page_number=page_data.get("page_number", 1),
            entities=page_entities,
            image_size=page_data.get("image_size", [0, 0]),
        ))

    return ExtractionResponse(
        request_id=request_id,
        entities=entities,
        pages=pages,
        metadata=result.get("metadata", {}),
        processing_time_ms=processing_time_ms,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(..., description="Document file to process"),
    extraction_mode: str = Form(
        "ensemble",
        description="Extraction mode: layoutlm, donut, or ensemble",
    ),
):
    """Upload a document and extract structured fields.

    Accepts PDF, PNG, JPG, TIFF, and BMP files. Returns extracted entities
    with labels, values, confidence scores, and bounding boxes.
    """
    _validate_upload(file)
    content = await _read_upload(file)
    request_id = str(uuid.uuid4())

    if extraction_mode not in ("layoutlm", "donut", "ensemble"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extraction_mode: {extraction_mode}. "
                   f"Must be one of: layoutlm, donut, ensemble",
        )

    logger.info(
        "Processing document: %s (size=%d bytes, mode=%s, request_id=%s)",
        file.filename, len(content), extraction_mode, request_id,
    )

    start_time = time.perf_counter()

    try:
        processor = get_processor()
        result = processor.process_document(
            source=content, extraction_mode=extraction_mode
        )
    except Exception as exc:
        logger.exception("Extraction failed for request %s", request_id)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return _build_response(result, request_id, elapsed_ms)


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(
    files: list[UploadFile] = File(
        ..., description="List of document files to process"
    ),
    extraction_mode: str = Form("ensemble"),
):
    """Batch-process multiple documents.

    Uploads and processes multiple files in a single request. Each file
    is validated and processed independently; failures on individual
    documents do not abort the entire batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch request.",
        )

    batch_request_id = str(uuid.uuid4())
    batch_start = time.perf_counter()

    results = []
    processor = get_processor()

    for idx, file in enumerate(files):
        file_request_id = f"{batch_request_id}-{idx}"
        try:
            _validate_upload(file)
            content = await _read_upload(file)

            start_time = time.perf_counter()
            result = processor.process_document(
                source=content, extraction_mode=extraction_mode
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            response = _build_response(result, file_request_id, elapsed_ms)
            results.append(response)

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Batch item %d failed: %s", idx, exc)
            results.append(ExtractionResponse(
                request_id=file_request_id,
                entities=[],
                pages=[],
                metadata={"error": str(exc), "filename": file.filename},
                processing_time_ms=0,
            ))

    total_elapsed = (time.perf_counter() - batch_start) * 1000

    return BatchExtractionResponse(
        request_id=batch_request_id,
        results=results,
        total_documents=len(files),
        total_processing_time_ms=total_elapsed,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status."""
    models_loaded = {
        "layoutlm": _processor is not None and _processor._layoutlm_extractor is not None,
        "donut": _processor is not None and _processor._donut_extractor is not None,
        "ocr": _processor is not None and _processor._ocr_pipeline is not None,
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded,
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
