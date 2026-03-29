import os
import json
import glob
from datetime import datetime

def get_latest_run_id(experiment_id="1", base_path="mlruns"):
    #find most recent run ID in mlruns folder
    experiment_path = os.path.join(base_path, experiment_id)
    if not os.path.exists(experiment_path):
        return None #if no runs exist, return nothing
    
    items = glob.glob(os.path.join(experiment_path, "*"))
    run_dirs = [d for d in items if os.path.isdir(d)]
    
    if not run_dirs:
        return None
    #find by modification time
    latest_run = max(run_dirs, key=os.path.getmtime)
    return os.path.basename(latest_run)

def generate_report(run_id = None, experiment_id = "1" ):
    
    if run_id is None:
        run_id = get_latest_run_id(experiment_id)
        if not run_id:
            print("No runs found")
            return
        #training data path
    metric_path = f"mlruns/{experiment_id}/{run_id}/artifacts/distilbert_metrics.json"
    output_dir = "artifacts/evaluation"
    report_file = os.path.join(output_dir, f"report_{run_id[:8]}.md")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if not os.path.exists(metric_path):
            print(f"Metrics file not found at {metric_path}")
            return
        with open(metric_path, "r") as f:
            metrics = json.load(f)
            #report generation formatting
            test_results = metrics.get("test_metrics", {})
            value_results = metrics.get("val_metrics")
            train_results = metrics.get("train_metrics")
        report_content = f"""Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
MLflow Run ID: `{run_id}`

Performance Summary
--------------------------------
Test Accuracy: {float(test_results.get('test_accuracy', 0)):.4f} 
Test F1 Macro: {float(test_results.get('test_f1_macro', 0)):.4f}
Test Loss:     {float(test_results.get('test_loss', 0)):.4f}

Training Health
--------------------------------
Validation Acc: {float(value_results.get('eval_accuracy', 0)):.4f}
Training Loss:  {float(train_results.get('train_loss', 0)):.4f}
Epochs:         {float(train_results.get('epoch', 0))}


NOTES:
Test Accuracy: percentage of documents the model correctly categorized
Test Loss: how wrong the model's predictions were, lower is better
F1 Macro: average of the F1 scores across the labels. Checks overall performance in all categories(governing law, non-compete, etc) '
Higher F1 = less hallucination.

Validation Acc: How well the model performed on the validation set during training, helps to check for overfitting(bad if the gap between accuracy and validation accuracy is large)
Training Loss: how wrong the model's predictions were during training, lower is better
Epochs: number of times the model goes through the full dataset
""" 
        with open(report_file, "w") as f:
            f.write(report_content)
        print(f"Report generated: {report_file}")
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    generate_report()