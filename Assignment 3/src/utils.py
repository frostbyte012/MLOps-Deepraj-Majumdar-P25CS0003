"""
utils.py - Utility functions shared across train / eval scripts.
"""

import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_metrics(pred) -> dict:
    """Compute accuracy for HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def save_evaluation_results(results: dict, path: str = "results/eval_results.json") -> None:
    """Persist evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {path}")


def load_evaluation_results(path: str = "results/eval_results.json") -> dict:
    """Load previously saved evaluation metrics."""
    with open(path) as f:
        return json.load(f)


def print_classification_report(true_labels: list, predicted_labels: list) -> str:
    """Print and return a full sklearn classification report."""
    report = classification_report(true_labels, predicted_labels)
    print(report)
    return report


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)