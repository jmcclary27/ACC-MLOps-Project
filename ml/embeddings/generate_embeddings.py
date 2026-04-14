from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from sentence_transformers import SentenceTransformer

from ml.data.text_helpers import chunk_legal_text_with_offsets, pdf_to_text


# -------------------------------------------------------------------
# Paths / Config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "retrieval_docs"
DEFAULT_OUTPUT_BASE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "embeddings"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EXPERIMENT_NAME = "contract-retrieval-embeddings"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CHUNK_CHARS = 350

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding_dim: int
    embedding_model: str


@dataclass(frozen=True)
class EmbeddingManifest:
    run_id: str
    created_at_utc: str
    embedding_model: str
    embedding_dim: int
    batch_size: int
    max_chunk_chars: int
    num_documents: int
    num_chunks: int
    input_paths: list[str]
    embeddings_path: str
    metadata_path: str
    config_path: str


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        logger.info("Extracting text from PDF: %s", path)
        return str(pdf_to_text(str(path), as_text=True))

    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        logger.info("Reading text file: %s", path)
        return read_text_file(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def collect_input_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]

    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported input files found under: {input_path}"
        )

    return files


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------
# Chunking / metadata creation
# -------------------------------------------------------------------

def build_chunk_metadata_for_document(
    path: Path,
    embedding_model_name: str,
    embedding_dim: int,
    max_chunk_chars: int,
) -> list[ChunkMetadata]:
    text = load_document_text(path)

    if not text or not text.strip():
        logger.warning("Skipping empty document: %s", path)
        return []

    raw_chunks = chunk_legal_text_with_offsets(
        text=text,
        max_chunk_chars=max_chunk_chars,
    )

    doc_id = path.stem
    rows: list[ChunkMetadata] = []

    for idx, chunk in enumerate(raw_chunks):
        chunk_text = str(chunk["text"]).strip()
        if not chunk_text:
            continue

        rows.append(
            ChunkMetadata(
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                source_path=str(path),
                text=chunk_text,
                start_char=int(chunk["start_char"]),
                end_char=int(chunk["end_char"]),
                chunk_index=idx,
                embedding_dim=embedding_dim,
                embedding_model=embedding_model_name,
            )
        )

    logger.info(
        "Built %d chunks for %s",
        len(rows),
        path.name,
    )
    return rows


# -------------------------------------------------------------------
# Embedding generation
# -------------------------------------------------------------------

def load_embedding_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


# -------------------------------------------------------------------
# MLflow helpers
# -------------------------------------------------------------------

def maybe_start_mlflow_run(
    experiment_name: str,
    run_name: str,
    enabled: bool,
):
    if not enabled:
        return None

    if mlflow is None:
        logger.warning("MLflow not installed, continuing without MLflow logging.")
        return None

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_mlflow_params_and_metrics(
    enabled: bool,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path,
) -> None:
    if not enabled or mlflow is None:
        return

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(artifacts_dir))


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_embedding_pipeline(
    input_path: Path,
    output_base_dir: Path,
    embedding_model_name: str,
    batch_size: int,
    max_chunk_chars: int,
    experiment_name: str,
    enable_mlflow: bool,
) -> Path:
    input_files = collect_input_files(input_path)
    run_id = f"embeddings_{utc_now_compact()}"
    run_output_dir = output_base_dir / run_id
    ensure_dir(run_output_dir)

    logger.info("Embedding run id: %s", run_id)
    logger.info("Output directory: %s", run_output_dir)

    mlflow_run = maybe_start_mlflow_run(
        experiment_name=experiment_name,
        run_name=run_id,
        enabled=enable_mlflow,
    )

    try:
        model = load_embedding_model(embedding_model_name)
        embedding_dim = int(model.get_sentence_embedding_dimension())

        all_metadata: list[ChunkMetadata] = []
        for path in input_files:
            doc_rows = build_chunk_metadata_for_document(
                path=path,
                embedding_model_name=embedding_model_name,
                embedding_dim=embedding_dim,
                max_chunk_chars=max_chunk_chars,
            )
            all_metadata.extend(doc_rows)

        if not all_metadata:
            raise RuntimeError("No chunk metadata was produced. Nothing to embed.")

        texts = [row.text for row in all_metadata]
        embeddings = generate_embeddings(
            model=model,
            texts=texts,
            batch_size=batch_size,
        )

        if embeddings.shape[0] != len(all_metadata):
            raise RuntimeError(
                f"Embedding row count mismatch: embeddings={embeddings.shape[0]}, "
                f"metadata={len(all_metadata)}"
            )

        embeddings_path = run_output_dir / "chunk_embeddings.npy"
        metadata_path = run_output_dir / "chunk_metadata.jsonl"
        config_path = run_output_dir / "config.json"
        manifest_path = run_output_dir / "manifest.json"

        np.save(embeddings_path, embeddings)

        metadata_rows = [asdict(row) for row in all_metadata]
        save_jsonl(metadata_path, metadata_rows)

        config_payload = {
            "input_path": str(input_path),
            "input_files": [str(p) for p in input_files],
            "embedding_model": embedding_model_name,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "max_chunk_chars": max_chunk_chars,
            "normalize_embeddings": True,
            "created_at_utc": utc_now_iso(),
        }
        save_json(config_path, config_payload)

        manifest = EmbeddingManifest(
            run_id=run_id,
            created_at_utc=utc_now_iso(),
            embedding_model=embedding_model_name,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            max_chunk_chars=max_chunk_chars,
            num_documents=len({row.doc_id for row in all_metadata}),
            num_chunks=len(all_metadata),
            input_paths=[str(p) for p in input_files],
            embeddings_path=str(embeddings_path),
            metadata_path=str(metadata_path),
            config_path=str(config_path),
        )
        save_json(manifest_path, asdict(manifest))

        logger.info("Saved embeddings to: %s", embeddings_path)
        logger.info("Saved metadata to: %s", metadata_path)
        logger.info("Saved manifest to: %s", manifest_path)

        params = {
            "embedding_model": embedding_model_name,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "max_chunk_chars": max_chunk_chars,
            "num_input_files": len(input_files),
            "input_path": str(input_path),
        }
        metrics = {
            "num_documents": float(len({row.doc_id for row in all_metadata})),
            "num_chunks": float(len(all_metadata)),
            "embedding_rows": float(embeddings.shape[0]),
            "embedding_dim": float(embeddings.shape[1]),
        }

        log_mlflow_params_and_metrics(
            enabled=enable_mlflow,
            params=params,
            metrics=metrics,
            artifacts_dir=run_output_dir,
        )

        return run_output_dir

    finally:
        if enable_mlflow and mlflow is not None and mlflow_run is not None:
            mlflow.end_run()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chunk embeddings for retrieval from contract documents."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Path to a single document or a directory of documents.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE_DIR,
        help="Base directory where versioned embedding artifacts will be saved.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help="Maximum characters per chunk passed into chunk_legal_text_with_offsets.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = run_embedding_pipeline(
        input_path=args.input_path,
        output_base_dir=args.output_base_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        max_chunk_chars=args.max_chunk_chars,
        experiment_name=args.experiment_name,
        enable_mlflow=not args.disable_mlflow,
    )

    print(f"Embedding artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()