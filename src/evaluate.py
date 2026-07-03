"""
Evaluation module for assessing model performance on the held-out test set.
Generates metrics, charts, threshold reports, and error analyses.
"""
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, Any
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Add parent directory to path to enable execution from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import ScreenRecaptureDataset


def plot_confusion_matrix(cm: np.ndarray, classes: list, output_path: Path) -> None:
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_pipeline(model_path: Path, test_data_dir: Path, results_dir: Path) -> None:
    """
    Evaluates the model on test data, performing error analysis and threshold search.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # 2. Prepare DataLoader
    print(f"Loading test dataset from {test_data_dir}...")
    test_wrapper = ScreenRecaptureDataset(
        data_dir=test_data_dir,
        image_size=config.IMAGE_SIZE,
        seed=config.RANDOM_SEED
    )
    
    test_ds = test_wrapper.get_dataset(
        batch_size=config.BATCH_SIZE,
        is_training=False,
        shuffle=False
    )
    
    # 3. Perform Inference
    print("Running inference...")
    y_prob = model.predict(test_ds, verbose=0).flatten()
    
    if getattr(config, "CALIBRATOR_PATH", None) and config.CALIBRATOR_PATH.exists():
        try:
            print(f"Applying Platt scaling calibrator from {config.CALIBRATOR_PATH}...")
            calibrator = joblib.load(config.CALIBRATOR_PATH)
            y_prob = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        except Exception as e:
            print(f"Warning: Failed to load and apply calibrator: {e}")

    y_true = np.array(test_wrapper.labels)
    image_paths = test_wrapper.image_paths
    
    # Ensure sizes match
    assert len(y_prob) == len(y_true) == len(image_paths), "Prediction length mismatch!"

    # 4. Threshold Evaluation
    print("Evaluating thresholds...")
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_threshold = 0.50
    best_f1 = -1.0
    threshold_data = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred_t)
        prec = precision_score(y_true, y_pred_t, zero_division=0)
        rec = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        
        cm_t = confusion_matrix(y_true, y_pred_t)
        # Avoid size mismatch errors if classes are missing in small dataset
        if cm_t.size == 4:
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
            fnr = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
        else:
            fpr, fnr = 0.0, 0.0
            
        threshold_data.append({
            "threshold": round(t, 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Save threshold metrics CSV
    threshold_df = pd.DataFrame(threshold_data)
    threshold_csv = results_dir / "threshold_metrics.csv"
    threshold_df.to_csv(threshold_csv, index=False)
    print(f"Saved threshold metrics to {threshold_csv}")
    print(f"Recommended threshold (highest F1-score): {best_threshold:.2f} (F1: {best_f1:.4f})")

    # 5. Compute Final Metrics at Recommended Threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case handler for single-class testing
        tn = cm[0, 0] if y_true[0] == 0 else 0
        tp = cm[0, 0] if y_true[0] == 1 else 0
        fp, fn = 0, 0

    metrics = {
        "recommended_threshold": float(best_threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "log_loss": float(loss),
        "mcc": float(mcc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

    # Save metrics JSON
    metrics_json_path = results_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics summary to {metrics_json_path}")

    # Display key items in console
    print("\n" + "="*40)
    print("Test Evaluation Summary")
    print("="*40)
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Log Loss:  {loss:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print("="*40)

    # 6. Save Plots
    # Confusion Matrix
    plot_confusion_matrix(cm, classes=["REAL", "SCREEN"], output_path=results_dir / "confusion_matrix.png")
    print(f"Generated Confusion Matrix plot: {results_dir / 'confusion_matrix.png'}")

    # ROC Curve
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=150)
    plt.close()
    print(f"Generated ROC curve: {results_dir / 'roc_curve.png'}")

    # Precision-Recall Curve
    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall_pr, precision_pr, color='blue', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"Generated Precision-Recall curve: {results_dir / 'precision_recall_curve.png'}")

    # Classification Report TXT
    report = classification_report(y_true, y_pred, target_names=["REAL", "SCREEN"], zero_division=0)
    report_path = results_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved classification report text to {report_path}")

    # Confidence Distribution Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_true == 0], bins=15, alpha=0.5, label='Real Photos (0)', color='blue')
    plt.hist(y_prob[y_true == 1], bins=15, alpha=0.5, label='Screen Photos (1)', color='red')
    plt.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({best_threshold:.2f})')
    plt.title('Confidence Distribution (Prediction Probabilities)')
    plt.xlabel('Probability of Screen Recapture')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "confidence_distribution.png", dpi=150)
    plt.close()
    print(f"Generated confidence distribution plot: {results_dir / 'confidence_distribution.png'}")

    # 7. Error Analysis
    print("Performing error analysis...")
    fp_dir = results_dir / "errors" / "false_positive"
    fn_dir = results_dir / "errors" / "false_negative"
    
    if fp_dir.exists():
        shutil.rmtree(fp_dir)
    if fn_dir.exists():
        shutil.rmtree(fn_dir)
        
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    error_records = []
    for path_str, prob, true, pred in zip(image_paths, y_prob, y_true, y_pred):
        path = Path(path_str)
        if true != pred:
            true_name = "SCREEN" if true == 1 else "REAL"
            pred_name = "SCREEN" if pred == 1 else "REAL"
            
            error_records.append({
                "filename": path.name,
                "true_label": true_name,
                "predicted_label": pred_name,
                "probability": round(float(prob), 4)
            })

            # Copy misclassified images to target subfolders
            if true == 0 and pred == 1:
                shutil.copy2(path, fp_dir / path.name)
            elif true == 1 and pred == 0:
                shutil.copy2(path, fn_dir / path.name)

    # Save error analysis CSV
    error_df = pd.DataFrame(error_records)
    error_csv = results_dir / "error_analysis.csv"
    error_df.to_csv(error_csv, index=False)
    print(f"Saved error analysis CSV to {error_csv} ({len(error_records)} errors logged)")


if __name__ == "__main__":
    evaluate_pipeline(
        model_path=config.CHECKPOINT_PATH,
        test_data_dir=config.DATASET_DIR / "test",
        results_dir=config.RESULTS_DIR
    )
