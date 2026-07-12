"""
train.py
========
Model training & comparison layer for the Cricket Analytics & Match
Prediction Platform.

This module trains five classification models on the engineered feature
matrix produced by `src/features/engineering.py`, evaluates each with
stratified k-fold cross-validation AND a held-out test set, and persists
the best-performing model (by cross-validated F1 score) along with its
metadata to `models/`.

Models compared:
    - Logistic Regression  (simple, interpretable linear baseline)
    - Decision Tree         (non-linear splits, easy to visualize)
    - Random Forest         (bagged ensemble, reduces overfitting)
    - Gradient Boosting     (sequential error-correcting ensemble)
    - XGBoost                (industry-standard gradient boosting)

Usage:
    from src.models.train import train_and_compare_models

    results = train_and_compare_models()

Or from the command line:
    python -m src.models.train
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _XGBOOST_AVAILABLE = False

from src.features.engineering import FEATURE_COLUMNS, build_model_dataset
from src.models.evaluate import evaluate_model
from src.utils.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    ensure_directories_exist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Scoring metrics computed during cross-validation for every candidate
# model. F1 is used as the primary model-selection criterion since match
# outcomes are close to balanced but F1 is more robust than raw accuracy
# if that balance drifts as more seasons are added.
CV_SCORING: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]
PRIMARY_METRIC: str = "f1"


def get_candidate_models() -> dict[str, object]:
    """
    Construct the five candidate model instances with sensible, documented
    hyperparameters. Kept as a function (rather than a module-level
    constant) so a fresh, unfitted instance of each model is created every
    time training runs.

    Returns:
        Dict mapping a human-readable model name to an unfitted
        scikit-learn / XGBoost estimator instance. XGBoost is omitted if
        the `xgboost` package is not installed in the current environment
        (logged as a warning rather than crashing the whole pipeline).
    """
    candidates: dict[str, object] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
        ),
    }
    if _XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        logger.warning(
            "xgboost is not installed in this environment -- skipping the "
            "XGBoost candidate model. Install it (`pip install xgboost`) to "
            "include it in the comparison."
        )
    return candidates


def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    """
    Run stratified k-fold cross-validation for a single model and
    summarize each scoring metric as a mean +/- std across folds.

    Stratified folds are used (rather than plain KFold) because match
    outcomes, while close to balanced overall, can be locally imbalanced
    within a fold by chance -- stratification keeps the win/loss ratio
    consistent across folds for a fairer comparison.

    Args:
        model: Unfitted scikit-learn-compatible estimator.
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Dict with keys like "cv_f1_mean", "cv_f1_std", "cv_accuracy_mean",
        etc. for every metric in `CV_SCORING`.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)

    summary: dict[str, float] = {}
    for metric in CV_SCORING:
        key = f"test_{metric}"
        summary[f"cv_{metric}_mean"] = float(scores[key].mean())
        summary[f"cv_{metric}_std"] = float(scores[key].std())
    return summary


def train_and_compare_models(save: bool = True, time_based_split: bool = True) -> dict:
    """
    End-to-end training pipeline: build features, split train/test,
    cross-validate and fit all candidate models, evaluate each on the
    held-out test set, select the best model by cross-validated F1 score,
    and persist the best model + run metadata to disk.

    Args:
        save: If True (default), persist the best model to
            `BEST_MODEL_PATH` and a JSON summary of every model's results
            to `MODEL_METADATA_PATH`.
        time_based_split: If True (default, changed 2026-07-10), the test
            set is the most recent `TEST_SIZE` fraction of matches in
            chronological order, rather than a random stratified sample.
            This is the more honest evaluation for a model meant to
            predict FUTURE matches: a random split can train on 2024 data
            and test on 2009 data, which doesn't reflect how the model
            would actually be used (predicting seasons it hasn't seen
            yet). Set to False to reproduce the old random-stratified-split
            behavior (e.g. for comparison, or if the dataset is too small
            for a clean chronological split to leave enough test examples).
            Note that `build_model_dataset()` already returns matches in
            chronological order (matches.csv is sorted by season/date
            during preprocessing), so a simple positional split is
            sufficient here -- no re-sorting needed.

    Returns:
        Dict with keys:
            "results": dict of per-model metrics (cv + test set), keyed by
                model name.
            "best_model_name": name of the selected model.
            "best_model": the fitted best estimator.
            "feature_columns": list of feature column names used.
            "test_set": (X_test, y_test) tuple, useful for downstream
                evaluation/visualization without retraining.
            "time_based_split": the value of `time_based_split` used for
                this run, recorded so downstream consumers (e.g. the
                Streamlit Model Performance page) know how to interpret
                the test metrics.

    Example:
        >>> run = train_and_compare_models()
        >>> run["best_model_name"] in {
        ...     "Logistic Regression", "Decision Tree", "Random Forest",
        ...     "Gradient Boosting", "XGBoost",
        ... }
        True
    """
    ensure_directories_exist()
    logger.info("Starting model training & comparison pipeline...")

    X, y, feature_names, _encoders = build_model_dataset(save=True)

    if time_based_split:
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(
            f"Time-based split: {X_train.shape[0]:,} train rows (earliest "
            f"matches), {X_test.shape[0]:,} test rows (most recent "
            f"matches, test_size={TEST_SIZE})."
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        logger.info(
            f"Random stratified split: {X_train.shape[0]:,} train rows, "
            f"{X_test.shape[0]:,} test rows (test_size={TEST_SIZE})."
        )

    candidates = get_candidate_models()
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for name, model in candidates.items():
        logger.info(f"Training and cross-validating: {name}...")
        start = time.time()

        cv_metrics = cross_validate_model(model, X_train, y_train)

        model.fit(X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        elapsed = time.time() - start
        logger.info(
            f"{name}: CV F1={cv_metrics['cv_f1_mean']:.3f} (+/-{cv_metrics['cv_f1_std']:.3f}) | "
            f"Test Accuracy={test_metrics['accuracy']:.3f} | "
            f"Test ROC-AUC={test_metrics['roc_auc']:.3f} | "
            f"trained in {elapsed:.1f}s"
        )

        fitted_models[name] = model
        results[name] = {
            **cv_metrics,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "training_seconds": round(elapsed, 2),
        }

    best_model_name = max(results, key=lambda name: results[name][f"cv_{PRIMARY_METRIC}_mean"])
    best_model = fitted_models[best_model_name]
    logger.info(
        f"Best model selected: {best_model_name} "
        f"(CV {PRIMARY_METRIC}={results[best_model_name][f'cv_{PRIMARY_METRIC}_mean']:.3f})"
    )

    if save:
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info(f"Saved best model ({best_model_name}) to {BEST_MODEL_PATH}")

        metadata = {
            "best_model_name": best_model_name,
            "primary_selection_metric": f"cv_{PRIMARY_METRIC}_mean",
            "feature_columns": feature_names,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "test_size": TEST_SIZE,
            "time_based_split": time_based_split,
            "cv_folds": CV_FOLDS,
            "random_seed": RANDOM_SEED,
            "results": results,
        }
        with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model comparison metadata to {MODEL_METADATA_PATH}")

        # Also persist EVERY fitted model (not just the best one) so the
        # Streamlit "Model Performance" page can show side-by-side
        # comparisons without retraining from scratch.
        all_models_path = MODELS_DIR / "all_models.joblib"
        joblib.dump(fitted_models, all_models_path)
        logger.info(f"Saved all fitted models to {all_models_path}")

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_columns": feature_names,
        "test_set": (X_test, y_test),
        "all_models": fitted_models,
        "time_based_split": time_based_split,
    }


if __name__ == "__main__":
    # Allows running `python -m src.models.train` directly to execute the
    # full training + comparison pipeline and persist the best model.
    run = train_and_compare_models()
    print(f"\nBest model: {run['best_model_name']}")
    print("\nModel comparison (cross-validated F1):")
    for name, metrics in sorted(
        run["results"].items(), key=lambda kv: kv[1]["cv_f1_mean"], reverse=True
    ):
        print(f"  {name:<22} CV F1={metrics['cv_f1_mean']:.3f}  Test Acc={metrics['test_accuracy']:.3f}  Test ROC-AUC={metrics['test_roc_auc']:.3f}")
