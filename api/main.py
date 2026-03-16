from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.inference import predict_document

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None


app = FastAPI(title="MLOps Model Server")

# Allow frontend dev server to call backend
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


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is healthy.",
    }


class DocumentRequest(BaseModel):
    text: str


@app.post("/clauses")
def process_clauses(request: DocumentRequest):
    """
    Accept raw text and return predictions.
    Useful for testing without file upload.
    """
    results = predict_document(request.text)
    return {"results": results}


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF is not installed. Add pymupdf to requirements.",
        )

    text_parts: list[str] = []
    pdf_stream = io.BytesIO(file_bytes)

    with fitz.open(stream=pdf_stream, filetype="pdf") as pdf:
        for page in pdf:
            text_parts.append(page.get_text())

    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    if Document is None:
        raise HTTPException(
            status_code=500,
            detail="python-docx is not installed. Add python-docx to requirements.",
        )

    doc_stream = io.BytesIO(file_bytes)
    doc = Document(doc_stream)
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
        detail="Unsupported file type. Please upload a .txt, .pdf, or .docx file.",
    )


@app.post("/upload-contract")
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a contract file, extract text, run inference, and return predictions.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extracted_text = extract_text(file.filename, file_bytes)

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded file.",
        )

    results = predict_document(extracted_text)

    return {
        "filename": file.filename,
        "num_predictions": len(results),
        "results": results,
    }