from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from ml.retrieval.search_faiss import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_BASE_DIR,
    DEFAULT_FAISS_BASE_DIR,
    load_embedding_model,
    load_faiss_index,
    load_metadata,
    resolve_embeddings_dir,
    resolve_faiss_dir,
    search_index,
)


# -------------------------------------------------------------------
# Paths / Config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DATASET_PATH = PROJECT_ROOT / "ml" / "retrieval" / "eval_dataset.json"
DEFAULT_EXPERIMENT_NAME = "contract-retrieval-eval"
DEFAULT_K_VALUES = [1, 3, 5, 10]


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_eval_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a list of query records.")

    required_keys = {"query_id", "query", "relevant_chunk_ids"}
    for row in data:
        if not required_keys.issubset(row.keys()):
            raise ValueError(
                f"Each evaluation row must contain keys {required_keys}. Got: {row}"
            )
        if not isinstance(row["relevant_chunk_ids"], list):
            raise ValueError("relevant_chunk_ids must be a list.")

    return data


def recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0

    top_k = retrieved_chunk_ids[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_chunk_ids)
    return hits / len(relevant_chunk_ids)


def precision_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0

    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_chunk_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def maybe_start_mlflow_run(experiment_name: str, run_name: str, enabled: bool):
    if not enabled:
        return None

    if mlflow is None:
        logger.warning("MLflow not installed, continuing without MLflow logging.")
        return None

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def maybe_log_mlflow(enabled: bool, params: dict[str, Any], metrics: dict[str, float]) -> None:
    if not enabled or mlflow is None:
        return

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate_queries(
    eval_rows: list[dict[str, Any]],
    faiss_input_path: Path,
    embeddings_input_path: Path,
    embedding_model_name: str,
    k_values: list[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    faiss_dir = resolve_faiss_dir(faiss_input_path)
    embeddings_dir = resolve_embeddings_dir(embeddings_input_path)

    logger.info("Using FAISS dir: %s", faiss_dir)
    logger.info("Using embeddings dir: %s", embeddings_dir)

    index = load_faiss_index(faiss_dir)
    metadata_rows = load_metadata(embeddings_dir)
    model = load_embedding_model(embedding_model_name)

    max_k = max(k_values)

    per_query_results: list[dict[str, Any]] = []

    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}
    precision_scores: dict[int, list[float]] = {k: [] for k in k_values}
    reciprocal_ranks: list[float] = []

    for row in eval_rows:
        query_id = str(row["query_id"])
        query = str(row["query"])
        relevant_chunk_ids = set(str(x) for x in row["relevant_chunk_ids"])

        results = search_index(
            query=query,
            index=index,
            metadata_rows=metadata_rows,
            model=model,
            top_k=max_k,
        )

        retrieved_chunk_ids = [result.chunk_id for result in results]

        query_metrics = {
            "query_id": query_id,
            "query": query,
            "relevant_chunk_ids": sorted(relevant_chunk_ids),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "reciprocal_rank": reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
        }

        reciprocal_ranks.append(query_metrics["reciprocal_rank"])

        for k in k_values:
            r_at_k = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
            p_at_k = precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)

            recall_scores[k].append(r_at_k)
            precision_scores[k].append(p_at_k)

            query_metrics[f"recall_at_{k}"] = r_at_k
            query_metrics[f"precision_at_{k}"] = p_at_k

        per_query_results.append(query_metrics)

    aggregate_metrics: dict[str, float] = {
        "num_queries": float(len(eval_rows)),
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }

    for k in k_values:
        aggregate_metrics[f"recall_at_{k}"] = mean(recall_scores[k]) if recall_scores[k] else 0.0
        aggregate_metrics[f"precision_at_{k}"] = mean(precision_scores[k]) if precision_scores[k] else 0.0

    return aggregate_metrics, per_query_results


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

def save_results_json(path: Path, metrics: dict[str, float], per_query_results: list[dict[str, Any]]) -> None:
    payload = {
        "aggregate_metrics": metrics,
        "per_query_results": per_query_results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation results to: %s", path)


def print_summary(metrics: dict[str, float]) -> None:
    print("\nRetrieval evaluation summary")
    print("---------------------------")
    for key, value in metrics.items():
        if key == "num_queries":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality with Recall@K, Precision@K, and MRR."
    )

    parser.add_argument(
        "--eval-dataset-path",
        type=Path,
        default=DEFAULT_EVAL_DATASET_PATH,
        help="Path to JSON evaluation dataset.",
    )
    parser.add_argument(
        "--faiss-input-path",
        type=Path,
        default=DEFAULT_FAISS_BASE_DIR,
        help="Path to specific FAISS run dir, or FAISS base dir.",
    )
    parser.add_argument(
        "--embeddings-input-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_BASE_DIR,
        help="Path to specific embeddings run dir, or embeddings base dir.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used to encode the query.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "ml" / "data" / "processed" / "retrieval_eval_results.json",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_rows = load_eval_dataset(args.eval_dataset_path)

    mlflow_run = maybe_start_mlflow_run(
        experiment_name=args.experiment_name,
        run_name="retrieval_eval",
        enabled=not args.disable_mlflow,
    )

    try:
        metrics, per_query_results = evaluate_queries(
            eval_rows=eval_rows,
            faiss_input_path=args.faiss_input_path,
            embeddings_input_path=args.embeddings_input_path,
            embedding_model_name=args.embedding_model,
            k_values=DEFAULT_K_VALUES,
        )

        print_summary(metrics)
        save_results_json(args.output_json, metrics, per_query_results)

        maybe_log_mlflow(
            enabled=not args.disable_mlflow,
            params={
                "eval_dataset_path": str(args.eval_dataset_path),
                "faiss_input_path": str(args.faiss_input_path),
                "embeddings_input_path": str(args.embeddings_input_path),
                "embedding_model": args.embedding_model,
            },
            metrics=metrics,
        )

    finally:
        if not args.disable_mlflow and mlflow is not None and mlflow_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()