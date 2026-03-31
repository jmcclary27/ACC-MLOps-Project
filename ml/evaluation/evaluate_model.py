# evaluate_model.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.model_registry import resolve_model_dir
from ml.training.dataset import ContractDataset


# -----------------------------
# Paths / Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "clean_chunk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

EVAL_JSON_PATH = OUTPUT_DIR / "distilbert_eval_metrics.json"
EVAL_PREDICTIONS_PATH = OUTPUT_DIR / "distilbert_eval_predictions.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "distilbert_confusion_matrix.csv"

MAX_LENGTH = 128
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

SELECTED_LABELS = {
    "Governing Law",
    "Termination For Convenience",
    "Exclusivity",
    "Non-Compete",
    "No-Solicit Of Employees",
    "IP Ownership Assignment",
    "Cap on Liability",
    "Change of Control",
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"text", "clause"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["text", "clause"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["clause"] = df["clause"].astype(str).str.strip()

    df = df[df["text"] != ""]
    df = df[df["clause"].isin(SELECTED_LABELS)].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering to selected labels.")

    return df


def safe_train_val_test_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        stratify=df["clause"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["clause"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_label_maps(label_map_path: Path):
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json not found at: {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    label2id = payload["label2id"]
    id2label = {int(k): v for k, v in payload["id2label"].items()}

    return label2id, id2label


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def predict_dataset(
    model,
    tokenizer,
    df: pd.DataFrame,
    label2id: dict,
    id2label: dict,
    batch_size: int = 16,
) -> pd.DataFrame:
    dataset = ContractDataset(
        texts=df["text"].tolist(),
        labels=df["clause"].tolist(),
        tokenizer=tokenizer,
        label2id=label2id,
        model_max_length=MAX_LENGTH,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    all_true_ids = []

    for start_idx in range(0, len(dataset), batch_size):
        batch_items = [dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(dataset)))]

        input_ids = torch.stack([item["input_ids"] for item in batch_items]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch_items]).to(device)
        labels = torch.stack([item["labels"] for item in batch_items]).cpu().numpy()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

        all_logits.append(logits)
        all_true_ids.extend(labels.tolist())

    logits = np.vstack(all_logits)
    probs = softmax(logits)
    pred_ids = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    true_labels = [id2label[int(i)] for i in all_true_ids]
    pred_labels = [id2label[int(i)] for i in pred_ids]

    pred_df = pd.DataFrame(
        {
            "text": df["text"].tolist(),
            "true_clause": true_labels,
            "predicted_clause": pred_labels,
            "confidence": confidences,
        }
    )

    return pred_df


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    print("Loading dataset...")
    df = load_and_validate_data(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    _, _, test_df = safe_train_val_test_split(df)
    print(f"Using test split of size: {len(test_df)}")

    model_dir = resolve_model_dir()
    label_map_path = model_dir / "label_map.json"

    print(f"Resolved model directory: {model_dir}")
    print("Loading label map...")
    label2id, id2label = load_label_maps(label_map_path)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    print("Running evaluation...")
    pred_df = predict_dataset(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        label2id=label2id,
        id2label=id2label,
    )

    y_true = pred_df["true_clause"].tolist()
    y_pred = pred_df["predicted_clause"].tolist()

    label_order = [id2label[i] for i in range(len(id2label))]

    metrics = {
        "resolved_model_dir": str(model_dir),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_order,
            output_dict=True,
            zero_division=0,
        ),
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)

    pred_df.to_csv(EVAL_PREDICTIONS_PATH, index=False)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    print(f"Metrics saved to: {EVAL_JSON_PATH}")
    print(f"Predictions saved to: {EVAL_PREDICTIONS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()