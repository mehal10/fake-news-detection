"""Sentiment-specific evaluation helpers (3-class metrics)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']


def evaluate_sentiment(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    # Prefer canonical order when available
    ordered = [l for l in SENTIMENT_LABELS if l in labels]
    for l in labels:
        if l not in ordered:
            ordered.append(l)
    labels = ordered

    acc = float(accuracy_score(y_true, y_pred))
    precision_w = float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    recall_w = float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    precision_m = float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    recall_m = float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )

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
        'precision': round(precision_w, 4),
        'recall': round(recall_w, 4),
        'f1_score': round(f1_w, 4),
        'precision_macro': round(precision_m, 4),
        'recall_macro': round(recall_m, 4),
        'f1_macro': round(f1_m, 4),
        'labels': labels,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'classification_report': report,
    }
