from __future__ import annotations

import io
import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.model_loader import load_model
from ml.inference import predict_document

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="MLOps Model Server")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DocumentRequest(BaseModel):
    text: str


def format_prediction_result(result: dict[str, object]) -> dict[str, object]:
    """
    Normalize a prediction result into the API response format.
    """
    return {
        "sentence": result["text"],
        "label": result["label"],
        "confidence": result["confidence"],
        "start_char": result["start_char"],
        "end_char": result["end_char"],
    }


def format_prediction_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return [format_prediction_result(result) for result in results]


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Return all expected API errors in a consistent format.
    """
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch unexpected server errors so clients never see raw stack traces.
    """
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )


@app.on_event("startup")
def startup_event() -> None:
    """
    Load the model once when the API starts.
    This avoids first-request latency and catches bad model paths early.
    """
    logger.info("Starting API and loading model...")
    load_model()
    logger.info("API startup complete.")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "API is healthy.",
    }


@app.post("/clauses")
def process_clauses(request: DocumentRequest) -> dict[str, object]:
    """
    Accept raw text and return model predictions.
    Useful for Swagger testing or direct frontend text testing.
    """
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    results = predict_document(clean_text)
    formatted_results = format_prediction_results(results)

    return {
        "source_type": "text",
        "num_predictions": len(formatted_results),
        "results": formatted_results,
    }


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF is not installed.",
        )

    text_parts: list[str] = []
    pdf_stream = io.BytesIO(file_bytes)

    try:
        with fitz.open(stream=pdf_stream, filetype="pdf") as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Failed to read PDF file.",
        ) from exc

    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    if Document is None:
        raise HTTPException(
            status_code=500,
            detail="python-docx is not installed.",
        )

    doc_stream = io.BytesIO(file_bytes)

    try:
        doc = Document(doc_stream)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Failed to read DOCX file.",
        ) from exc

    return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    lowered = filename.lower()

    if lowered.endswith(".txt"):
        return extract_text_from_txt(file_bytes)

    if lowered.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    if lowered.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    raise HTTPException(
        status_code=400,
        detail="Invalid file type. Please upload a .txt, .pdf, or .docx file.",
    )


@app.post("/upload-contract")
async def upload_contract(file: UploadFile = File(...)) -> dict[str, object]:
    """
    Upload a contract file, extract its text, run inference, and return results.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is missing a filename.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    extracted_text = extract_text(file.filename, file_bytes)

    if not extracted_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded file.",
        )

    results = predict_document(extracted_text)
    formatted_results = format_prediction_results(results)

    return {
        "source_type": "file",
        "filename": file.filename,
        "extracted_text": extracted_text,
        "extracted_text_length": len(extracted_text),
        "num_predictions": len(formatted_results),
        "results": formatted_results,
    }