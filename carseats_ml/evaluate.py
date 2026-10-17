"""Evaluation helpers: metrics, confusion matrices and feature importances."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

#: Human-readable labels for the binary target.
CLASS_LABELS = ["Low (0)", "High (1)"]


def confusion(y_true, y_pred):
    """Confusion matrix for the two classes."""
    return confusion_matrix(y_true, y_pred)


def report(y_true, y_pred, labels: list[str] | None = None) -> str:
    """Text classification report with named classes."""
    return classification_report(y_true, y_pred, target_names=labels or CLASS_LABELS)


def importances(model, feature_names) -> pd.DataFrame:
    """Feature importances from a fitted tree model, most important first."""
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def comparison_table(scores: dict[str, float]) -> pd.DataFrame:
    """Rank models by accuracy, best first."""
    table = pd.DataFrame({"model": list(scores), "accuracy": list(scores.values())}).sort_values(
        "accuracy", ascending=False
    )
    return table.reset_index(drop=True)


def to_markdown(scores: dict[str, float]) -> str:
    """Markdown table of model accuracies, best first."""
    table = comparison_table(scores)
    lines = ["| Model | Accuracy |", "| --- | ---: |"]
    lines += [f"| {row.model} | {row.accuracy:.3f} |" for row in table.itertuples()]
    return "\n".join(lines)
