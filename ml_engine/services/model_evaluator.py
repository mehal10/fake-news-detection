"""Evaluation metrics for classification models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1, confusion matrix, and full report.
    Uses weighted averages for multi-class; binary-friendly defaults.
    """
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )

    # Per-class metrics
    per_class = {}
    for lab in labels:
        if lab in report:
            per_class[lab] = {
                'precision': round(report[lab]['precision'], 4),
                'recall': round(report[lab]['recall'], 4),
                'f1': round(report[lab]['f1-score'], 4),
                'support': int(report[lab]['support']),
            }

    return {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'labels': labels,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'classification_report': report,
    }
