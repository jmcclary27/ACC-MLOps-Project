import json
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ml.model_registry import (
    ModelVersionInfo,
    get_next_version,
    get_versioned_model_dir,
    mark_as_production,
    save_version_metadata,
    utc_now_iso,
)
from ml.training.dataset import ContractDataset


# -----------------------------
# Paths / Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "clean_chunk_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
METRICS_OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
METRICS_JSON_PATH = METRICS_OUTPUT_DIR / "distilbert_metrics.json"
PREDICTIONS_CSV_PATH = METRICS_OUTPUT_DIR / "distilbert_test_predictions.csv"

MODEL_NAME = "distilbert-base-uncased"
MODEL_BASENAME = "distilbert_clause_classifier"
MAX_LENGTH = 128

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

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

NUM_EPOCHS = 6
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

MLFLOW_EXPERIMENT_NAME = "contract-clause-classifier"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def print_distribution(name: str, labels: pd.Series) -> None:
    print(f"\n{name} label distribution:")
    print(labels.value_counts())


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

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_label_maps(labels: pd.Series):
    unique_labels = sorted(labels.unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_metrics_factory(id2label: dict):
    label_names_in_order = [id2label[i] for i in range(len(id2label))]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

        per_class_f1 = f1_score(
            labels,
            preds,
            average=None,
            labels=list(range(len(label_names_in_order))),
            zero_division=0,
        )

        for label_name, f1_val in zip(label_names_in_order, per_class_f1):
            metric_key = f"f1_{label_name.lower().replace(' ', '_').replace('-', '_')}"
            metrics[metric_key] = float(f1_val)

        return metrics

    return compute_metrics


def save_label_map(output_dir: Path, label2id: dict, id2label: dict):
    label_map_path = output_dir / "label_map.json"
    payload = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return label_map_path


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    model_version = get_next_version()
    model_output_dir = get_versioned_model_dir(model_version)
    label_map_path = model_output_dir / "label_map.json"
    checkpoints_dir = model_output_dir / "checkpoints"

    print("Loading dataset...")
    df = load_and_validate_data(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    print_distribution("Full dataset", df["clause"])

    train_df, val_df, test_df = safe_train_val_test_split(df)

    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print_distribution("Train", train_df["clause"])
    print_distribution("Validation", val_df["clause"])
    print_distribution("Test", test_df["clause"])

    label2id, id2label = build_label_maps(df["clause"])
    num_labels = len(label2id)

    print("\nLabel mapping:")
    print(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ContractDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["clause"].tolist(),
        tokenizer=tokenizer,
        label2id=label2id,
        model_max_length=MAX_LENGTH,
    )
    val_dataset = ContractDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["clause"].tolist(),
        tokenizer=tokenizer,
        label2id=label2id,
        model_max_length=MAX_LENGTH,
    )
    test_dataset = ContractDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["clause"].tolist(),
        tokenizer=tokenizer,
        label2id=label2id,
        model_max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
        dataloader_num_workers=0,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_factory(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    with mlflow.start_run(run_name="distilbert_sentence_classifier") as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_basename", MODEL_BASENAME)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_output_dir", str(model_output_dir))
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_batch_size", TRAIN_BATCH_SIZE)
        mlflow.log_param("eval_batch_size", EVAL_BATCH_SIZE)
        mlflow.log_param("num_train_epochs", NUM_EPOCHS)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("early_stopping_patience", EARLY_STOPPING_PATIENCE)
        mlflow.log_param("selected_labels", sorted(list(SELECTED_LABELS)))

        print("\nStarting training...")
        train_result = trainer.train()

        train_metrics = train_result.metrics
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"train_{key}", value)

        print("\nEvaluating on validation set...")
        val_metrics = trainer.evaluate(eval_dataset=val_dataset)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"val_{key}", value)

        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        predictions_output = trainer.predict(test_dataset)
        test_logits = predictions_output.predictions
        test_pred_ids = np.argmax(test_logits, axis=-1)
        test_true_ids = predictions_output.label_ids

        test_pred_labels = [id2label[int(i)] for i in test_pred_ids]
        test_true_labels = [id2label[int(i)] for i in test_true_ids]

        pred_df = pd.DataFrame(
            {
                "text": test_df["text"].tolist(),
                "true_clause": test_true_labels,
                "predicted_clause": test_pred_labels,
            }
        )
        pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

        report_dict = classification_report(
            test_true_labels,
            test_pred_labels,
            output_dict=True,
            zero_division=0,
        )

        all_metrics = {
            "model_version": model_version,
            "model_output_dir": str(model_output_dir),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "classification_report": report_dict,
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        }

        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        print("\nSaving best model...")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_output_dir))
        tokenizer.save_pretrained(str(model_output_dir))
        save_label_map(model_output_dir, label2id, id2label)

        version_info = ModelVersionInfo(
            model_name=MODEL_BASENAME,
            version=model_version,
            artifact_path=str(model_output_dir),
            created_at=utc_now_iso(),
            stage="staging",
        )
        version_metadata_path = save_version_metadata(version_info)
        production_pointer_path = mark_as_production(version_info)

        mlflow.set_tag("model_stage", "production")
        mlflow.set_tag("production_model_version", str(model_version))

        mlflow.log_artifact(str(METRICS_JSON_PATH))
        mlflow.log_artifact(str(PREDICTIONS_CSV_PATH))
        mlflow.log_artifact(str(label_map_path))
        mlflow.log_artifact(str(version_metadata_path))
        mlflow.log_artifact(str(production_pointer_path))

        try:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                    "tokenizer": tokenizer,
                },
                artifact_path="model",
            )
        except Exception as e:
            print(f"Skipping mlflow.transformers.log_model due to: {e}")

        print("\nDone.")
        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Versioned model saved to: {model_output_dir}")
        print(f"Production pointer saved to: {production_pointer_path}")
        print(f"Metrics saved to: {METRICS_JSON_PATH}")
        print(f"Predictions saved to: {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()