"""
evaluate.py
===========
Model evaluation layer for the Cricket Analytics & Match Prediction
Platform.

This module is intentionally separate from `src/models/train.py`: training
decides WHICH model wins and persists it, while this module defines HOW any
single model is scored. Keeping evaluation logic here means the exact same
metric computation is used in three places without duplication:
    1. Inside the cross-model comparison in `train.py`.
    2. Re-evaluating a previously saved model later (e.g. on new data)
       without retraining everything.
    3. The Streamlit "Model Performance" page, which re-displays these
       same metrics interactively.

Usage:
    from src.models.evaluate import evaluate_model, get_classification_report

    metrics = evaluate_model(fitted_model, X_test, y_test)
    print(metrics["accuracy"], metrics["roc_auc"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute a standard set of classification metrics for a fitted model on
    a held-out test set.

    Args:
        model: A fitted scikit-learn-compatible classifier exposing both
            `.predict()` and `.predict_proba()`.
        X_test: Held-out feature matrix.
        y_test: Held-out true target labels (1 = team1 won, 0 = team2 won).

    Returns:
        Dict with keys:
            "accuracy", "precision", "recall", "f1", "roc_auc" (floats),
            "confusion_matrix" (2x2 nested list: [[TN, FP], [FN, TP]]),
            "y_pred" (np.ndarray of predicted labels),
            "y_proba" (np.ndarray of predicted positive-class probabilities).

    Example:
        >>> metrics = evaluate_model(fitted_model, X_test, y_test)
        >>> 0.0 <= metrics["roc_auc"] <= 1.0
        True
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def get_classification_report(model, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    Produce scikit-learn's full text classification report (per-class
    precision/recall/F1/support) for a fitted model on a held-out test set.

    Args:
        model: A fitted scikit-learn-compatible classifier.
        X_test: Held-out feature matrix.
        y_test: Held-out true target labels.

    Returns:
        Multi-line formatted string, suitable for printing or displaying
        verbatim in the Streamlit dashboard inside a code block.
    """
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"], zero_division=0
    )


def get_roc_curve_points(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, np.ndarray]:
    """
    Compute the false-positive-rate / true-positive-rate points needed to
    plot an ROC curve for a fitted model, plus the area under it.

    Args:
        model: A fitted scikit-learn-compatible classifier exposing
            `.predict_proba()`.
        X_test: Held-out feature matrix.
        y_test: Held-out true target labels.

    Returns:
        Dict with keys "fpr", "tpr", "thresholds" (each a 1-D np.ndarray)
        and "auc" (float), ready to feed directly into a Plotly line chart.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = float(roc_auc_score(y_test, y_proba))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importances (tree-based models) or absolute
    coefficient magnitudes (linear models) from a fitted model, normalized
    to sum to 1 for easy cross-model comparison.

    Args:
        model: A fitted scikit-learn-compatible estimator. Must expose
            either `.feature_importances_` (tree-based models) or
            `.coef_` (linear models such as LogisticRegression).
        feature_names: Column names corresponding to the model's input
            features, in the same order used during training.

    Returns:
        DataFrame with columns "feature" and "importance", sorted by
        importance descending. Returns an empty DataFrame with a logged
        warning if the model exposes neither attribute (no silent failure).
    """
    # Unwrap a scikit-learn Pipeline (e.g. the scaled Logistic Regression)
    # to get at the underlying estimator's importances/coefficients.
    inner_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(inner_model, "feature_importances_"):
        raw_importances = np.asarray(inner_model.feature_importances_)
    elif hasattr(inner_model, "coef_"):
        raw_importances = np.abs(np.asarray(inner_model.coef_)).flatten()
    else:
        logger.warning(
            f"Model {type(inner_model).__name__} exposes neither "
            f"`feature_importances_` nor `coef_` -- cannot compute "
            f"feature importance."
        )
        return pd.DataFrame(columns=["feature", "importance"])

    total = raw_importances.sum()
    normalized = raw_importances / total if total > 0 else raw_importances

    importance_df = pd.DataFrame({"feature": feature_names, "importance": normalized})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df
