from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required for this script. Install faiss-cpu first."
    ) from exc

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


# -------------------------------------------------------------------
# Paths / Config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDINGS_BASE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "embeddings"
DEFAULT_INDEX_BASE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "faiss"

DEFAULT_EXPERIMENT_NAME = "contract-retrieval-faiss"
DEFAULT_INDEX_TYPE = "flat_ip"  # cosine similarity if embeddings are normalized


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
class FaissManifest:
    run_id: str
    created_at_utc: str
    source_embeddings_dir: str
    embeddings_path: str
    metadata_path: str
    config_path: str
    index_path: str
    manifest_path: str
    embedding_dim: int
    num_vectors: int
    index_type: str


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def resolve_latest_embeddings_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Embeddings base directory does not exist: {base_dir}")

    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No embedding runs found under: {base_dir}")

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def resolve_embeddings_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        required = [
            input_path / "chunk_embeddings.npy",
            input_path / "chunk_metadata.jsonl",
            input_path / "config.json",
        ]
        if all(p.exists() for p in required):
            return input_path

        return resolve_latest_embeddings_dir(input_path)

    raise ValueError(
        "Embeddings input path must be a directory containing embedding artifacts, "
        "or a base directory containing versioned embedding runs."
    )


def load_embedding_artifacts(embeddings_dir: Path) -> tuple[np.ndarray, Path, Path, Path]:
    embeddings_path = embeddings_dir / "chunk_embeddings.npy"
    metadata_path = embeddings_dir / "chunk_metadata.jsonl"
    config_path = embeddings_dir / "config.json"

    for path in [embeddings_path, metadata_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D array, got shape {embeddings.shape}"
        )

    embeddings = embeddings.astype(np.float32, copy=False)
    return embeddings, embeddings_path, metadata_path, config_path


# -------------------------------------------------------------------
# FAISS helpers
# -------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, index_type: str) -> faiss.Index:
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build FAISS index with zero vectors.")

    dim = int(embeddings.shape[1])

    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "flat_l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(
            f"Unsupported index_type={index_type}. Use 'flat_ip' or 'flat_l2'."
        )

    index.add(embeddings)
    return index


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

def run_faiss_build(
    embeddings_input_path: Path,
    output_base_dir: Path,
    index_type: str,
    experiment_name: str,
    enable_mlflow: bool,
) -> Path:
    embeddings_dir = resolve_embeddings_dir(embeddings_input_path)
    run_id = f"faiss_{utc_now_compact()}"
    output_dir = output_base_dir / run_id
    ensure_dir(output_dir)

    logger.info("Using embeddings dir: %s", embeddings_dir)
    logger.info("FAISS run id: %s", run_id)
    logger.info("Output dir: %s", output_dir)

    mlflow_run = maybe_start_mlflow_run(
        experiment_name=experiment_name,
        run_name=run_id,
        enabled=enable_mlflow,
    )

    try:
        embeddings, embeddings_path, metadata_path, config_path = load_embedding_artifacts(
            embeddings_dir
        )
        config = read_json(config_path)
        metadata_count = count_jsonl_rows(metadata_path)

        if metadata_count != embeddings.shape[0]:
            raise RuntimeError(
                "Metadata row count does not match embedding rows: "
                f"{metadata_count} != {embeddings.shape[0]}"
            )

        embedding_dim = int(embeddings.shape[1])
        num_vectors = int(embeddings.shape[0])

        index = build_faiss_index(embeddings, index_type=index_type)

        index_path = output_dir / "faiss.index"
        manifest_path = output_dir / "manifest.json"
        build_config_path = output_dir / "config.json"

        faiss.write_index(index, str(index_path))

        build_config = {
            "created_at_utc": utc_now_iso(),
            "source_embeddings_dir": str(embeddings_dir),
            "source_embeddings_path": str(embeddings_path),
            "source_metadata_path": str(metadata_path),
            "source_config_path": str(config_path),
            "source_embedding_model": config.get("embedding_model"),
            "source_embedding_dim": config.get("embedding_dim"),
            "index_type": index_type,
            "num_vectors": num_vectors,
            "embedding_dim": embedding_dim,
        }
        save_json(build_config_path, build_config)

        manifest = FaissManifest(
            run_id=run_id,
            created_at_utc=utc_now_iso(),
            source_embeddings_dir=str(embeddings_dir),
            embeddings_path=str(embeddings_path),
            metadata_path=str(metadata_path),
            config_path=str(config_path),
            index_path=str(index_path),
            manifest_path=str(manifest_path),
            embedding_dim=embedding_dim,
            num_vectors=num_vectors,
            index_type=index_type,
        )
        save_json(manifest_path, asdict(manifest))

        logger.info("Saved FAISS index to: %s", index_path)
        logger.info("Saved manifest to: %s", manifest_path)

        params = {
            "source_embeddings_dir": str(embeddings_dir),
            "source_embedding_model": config.get("embedding_model"),
            "source_embedding_dim": embedding_dim,
            "index_type": index_type,
        }
        metrics = {
            "num_vectors": float(num_vectors),
            "embedding_dim": float(embedding_dim),
            "metadata_rows": float(metadata_count),
        }

        log_mlflow_params_and_metrics(
            enabled=enable_mlflow,
            params=params,
            metrics=metrics,
            artifacts_dir=output_dir,
        )

        return output_dir

    finally:
        if enable_mlflow and mlflow is not None and mlflow_run is not None:
            mlflow.end_run()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from saved chunk embeddings."
    )

    parser.add_argument(
        "--embeddings-input-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_BASE_DIR,
        help=(
            "Path to a specific embeddings run directory, or the base embeddings "
            "directory containing versioned runs."
        ),
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=DEFAULT_INDEX_BASE_DIR,
        help="Base directory where versioned FAISS artifacts will be saved.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default=DEFAULT_INDEX_TYPE,
        choices=["flat_ip", "flat_l2"],
        help=(
            "FAISS index type. Use flat_ip for normalized embeddings / cosine-style "
            "retrieval, or flat_l2 for Euclidean distance."
        ),
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

    output_dir = run_faiss_build(
        embeddings_input_path=args.embeddings_input_path,
        output_base_dir=args.output_base_dir,
        index_type=args.index_type,
        experiment_name=args.experiment_name,
        enable_mlflow=not args.disable_mlflow,
    )

    print(f"FAISS artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()