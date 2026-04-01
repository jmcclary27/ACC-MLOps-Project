# export_hard_examples.py
import json
from pathlib import Path

import pandas as pd


# -----------------------------
# Paths / Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "distilbert_eval_predictions.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "hard_examples.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "hard_examples.csv"

TOP_N_HIGH_CONF_WRONG = 100
TOP_N_LOW_CONF_CORRECT = 100


# -----------------------------
# Helpers
# -----------------------------
def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found at: {predictions_path}\n"
            "Run evaluate_model.py first."
        )

    df = pd.read_csv(predictions_path)

    required_columns = {"text", "true_clause", "predicted_clause", "confidence"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df["text"].astype(str)
    df["true_clause"] = df["true_clause"].astype(str)
    df["predicted_clause"] = df["predicted_clause"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = df.dropna(subset=["confidence"]).copy()
    df["is_correct"] = df["true_clause"] == df["predicted_clause"]

    return df


def row_to_record(row: pd.Series) -> dict:
    return {
        "text": row["text"],
        "true_clause": row["true_clause"],
        "predicted_clause": row["predicted_clause"],
        "confidence": float(row["confidence"]),
        "is_correct": bool(row["is_correct"]),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading evaluation predictions...")
    df = load_predictions(PREDICTIONS_PATH)

    wrong_df = df[~df["is_correct"]].copy()
    correct_df = df[df["is_correct"]].copy()

    high_conf_wrong = wrong_df.sort_values("confidence", ascending=False).head(TOP_N_HIGH_CONF_WRONG)
    low_conf_correct = correct_df.sort_values("confidence", ascending=True).head(TOP_N_LOW_CONF_CORRECT)

    payload = {
        "summary": {
            "total_examples": int(len(df)),
            "num_correct": int(correct_df.shape[0]),
            "num_wrong": int(wrong_df.shape[0]),
            "accuracy": float(correct_df.shape[0] / len(df)) if len(df) > 0 else 0.0,
            "top_n_high_conf_wrong": TOP_N_HIGH_CONF_WRONG,
            "top_n_low_conf_correct": TOP_N_LOW_CONF_CORRECT,
        },
        "high_confidence_wrong": [row_to_record(row) for _, row in high_conf_wrong.iterrows()],
        "low_confidence_correct": [row_to_record(row) for _, row in low_conf_correct.iterrows()],
    }

    review_df = pd.concat(
        [
            high_conf_wrong.assign(review_bucket="high_confidence_wrong"),
            low_conf_correct.assign(review_bucket="low_confidence_correct"),
        ],
        ignore_index=True,
    )

    review_df = review_df[
        ["review_bucket", "text", "true_clause", "predicted_clause", "confidence", "is_correct"]
    ]

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    review_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\nHard example export complete.")
    print(f"JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"CSV saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()