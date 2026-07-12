"""
live_win_probability.py
========================
Model training AND prediction layer for the IN-MATCH ("live") win-
probability model -- the counterpart to `src/models/train.py` +
`src/models/predict.py` for the PRE-MATCH model.

See `src/features/live_state_engineering.py`'s module docstring for the
full rationale (short version: pre-match prediction has a low,
well-documented ceiling for T20 cricket -- see docs/MODEL_CARD.md -- so
this module instead predicts win probability from ball-by-ball match
state during a second-innings chase, a much better-posed problem).

------------------------------------------------------------------------
CRITICAL: MATCH-LEVEL CROSS-VALIDATION AND SPLITTING
------------------------------------------------------------------------
Every ball in a match shares that match's outcome. Both the
cross-validation folds AND the train/test split MUST be grouped by
`match_id` -- never split at the ball level -- or the model would be
evaluated on balls from matches it already saw other balls of during
training, inflating every reported metric in a way that would not
reflect genuine out-of-match generalization. This module uses
`sklearn.model_selection.GroupKFold` (grouped by match_id) for
cross-validation, and a match-level chronological split (mirroring
`src.models.train`'s `time_based_split`, but splitting whole MATCHES,
each match's balls staying together) for the held-out test set.

Usage (training):
    from src.models.live_win_probability import train_and_compare_live_models

    run = train_and_compare_live_models()

Or from the command line:
    python -m src.models.live_win_probability

Usage (prediction, for a single hypothetical match state):
    from src.models.live_win_probability import predict_live_win_probability

    result = predict_live_win_probability(
        batting_team="Mumbai Indians", bowling_team="Chennai Super Kings",
        venue="Wankhede Stadium", target=180, runs_scored=90,
        wickets_lost=3, balls_bowled=60,
    )
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _XGBOOST_AVAILABLE = False

from src.features.live_state_engineering import (
    BALLS_PER_OVER,
    FEATURE_COLUMNS,
    TOTAL_LEGAL_BALLS,
    build_live_state_dataset,
    get_match_state_timeline,
)
from src.models.evaluate import evaluate_model
from src.utils.config import (
    BEST_LIVE_MODEL_PATH,
    CV_FOLDS,
    LIVE_ENCODERS_PATH,
    LIVE_MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    ensure_directories_exist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

CV_SCORING: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]
PRIMARY_METRIC: str = "f1"


class LiveModelNotTrainedError(Exception):
    """
    Raised when `predict_live_win_probability()` (or any function needing
    the persisted live model) is called before `python -m
    src.models.live_win_probability` has been run at least once.
    """
    pass


def get_candidate_models() -> dict[str, object]:
    """
    Construct candidate model instances for the live win-probability task.

    Unlike the pre-match model, the live-state dataset is large (~130,000
    balls) and its features are dense, well-behaved numeric signals (score,
    wickets, overs, run rates) rather than mostly-categorical team/venue
    identity -- ensemble tree models are expected to do noticeably better
    here than on the pre-match task, since there's a genuine non-linear
    relationship (e.g. "10 needed off 6 balls" is a very different
    situation than "10 needed off 30 balls," not a smooth linear one) for
    them to actually capture.

    Returns:
        Dict mapping model name to an unfitted estimator. XGBoost is
        omitted if the `xgboost` package isn't installed (logged, not fatal).
    """
    candidates: dict[str, object] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=15,
            random_state=RANDOM_SEED, n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED,
        ),
    }
    if _XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=1,
        )
    else:
        logger.warning(
            "xgboost is not installed in this environment -- skipping the "
            "XGBoost candidate model for the live win-probability comparison."
        )
    return candidates


def cross_validate_live_model(model, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series) -> dict[str, float]:
    """
    Run GROUPED k-fold cross-validation (grouped by match_id) for a single
    model. See module docstring: this MUST be grouped, not a plain/
    stratified fold, or balls from the same match would leak across folds.

    Args:
        model: Unfitted scikit-learn-compatible estimator.
        X_train: Training feature matrix (ball-level rows).
        y_train: Training target vector.
        groups_train: match_id per row, aligned with X_train/y_train --
            defines which rows are NOT allowed to split across folds.

    Returns:
        Dict with keys like "cv_f1_mean", "cv_f1_std", etc. for every
        metric in `CV_SCORING`.
    """
    cv = GroupKFold(n_splits=CV_FOLDS)
    scores = cross_validate(
        model, X_train, y_train, cv=cv, groups=groups_train, scoring=CV_SCORING, n_jobs=1
    )

    summary: dict[str, float] = {}
    for metric in CV_SCORING:
        key = f"test_{metric}"
        summary[f"cv_{metric}_mean"] = float(scores[key].mean())
        summary[f"cv_{metric}_std"] = float(scores[key].std())
    return summary


def _match_level_time_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple:
    """
    Split ball-level rows into train/test sets by MATCH, chronologically
    (earliest ~80% of matches -> train, most recent ~20% -> test), so that
    every ball from a given match lands entirely on one side of the split.

    `groups` (match_id) already reflects chronological order at the match
    level, since `build_live_state_dataset()` is built from
    `_build_ball_state()`, which itself derives from `deliveries_clean`
    sorted by `(match_id, over, ball)` where `match_id` was assigned during
    `preprocess.py`'s chronological sort -- so a simple ordinal split over
    the UNIQUE, in-first-appearance-order match_id sequence reproduces
    chronological order without needing to re-merge in match dates here.

    Args:
        X: Full ball-level feature matrix.
        y: Full ball-level target vector.
        groups: match_id per row, aligned with X/y.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, groups_train).
    """
    unique_matches_in_order = groups.drop_duplicates().tolist()
    split_point = int(len(unique_matches_in_order) * (1 - TEST_SIZE))
    train_match_ids = set(unique_matches_in_order[:split_point])

    train_mask = groups.isin(train_match_ids)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    groups_train = groups[train_mask]

    logger.info(
        f"Match-level time-based split: {len(train_match_ids):,} train "
        f"matches ({X_train.shape[0]:,} balls), "
        f"{len(unique_matches_in_order) - len(train_match_ids):,} test "
        f"matches ({X_test.shape[0]:,} balls)."
    )
    return X_train, X_test, y_train, y_test, groups_train


def train_and_compare_live_models(save: bool = True) -> dict:
    """
    End-to-end training pipeline for the live win-probability model:
    build ball-level features, split train/test at the MATCH level
    (chronologically), cross-validate (grouped by match) and fit all
    candidate models, evaluate each on the held-out test matches, select
    the best by cross-validated F1, and persist to disk.

    Args:
        save: If True (default), persist the best model to
            `BEST_LIVE_MODEL_PATH` and a JSON summary to
            `LIVE_MODEL_METADATA_PATH`.

    Returns:
        Dict with keys: "results", "best_model_name", "best_model",
        "feature_columns", "test_set" (X_test, y_test), "all_models".
    """
    ensure_directories_exist()
    logger.info("Starting live win-probability model training & comparison pipeline...")

    X, y, feature_names, _encoders, groups = build_live_state_dataset(save=True)
    X_train, X_test, y_train, y_test, groups_train = _match_level_time_split(X, y, groups)

    candidates = get_candidate_models()
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for name, model in candidates.items():
        logger.info(f"Training and cross-validating (grouped by match): {name}...")
        start = time.time()

        cv_metrics = cross_validate_live_model(model, X_train, y_train, groups_train)

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
        f"Best live model selected: {best_model_name} "
        f"(CV {PRIMARY_METRIC}={results[best_model_name][f'cv_{PRIMARY_METRIC}_mean']:.3f})"
    )

    if save:
        joblib.dump(best_model, BEST_LIVE_MODEL_PATH)
        logger.info(f"Saved best live model ({best_model_name}) to {BEST_LIVE_MODEL_PATH}")

        metadata = {
            "best_model_name": best_model_name,
            "primary_selection_metric": f"cv_{PRIMARY_METRIC}_mean",
            "feature_columns": feature_names,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "train_matches": int(groups_train.nunique()),
            "test_matches": int(groups[~groups.isin(set(groups_train))].nunique()),
            "test_size": TEST_SIZE,
            "split_strategy": "match-level chronological (see module docstring)",
            "cv_strategy": "GroupKFold by match_id (see module docstring)",
            "cv_folds": CV_FOLDS,
            "random_seed": RANDOM_SEED,
            "results": results,
        }
        with open(LIVE_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved live model comparison metadata to {LIVE_MODEL_METADATA_PATH}")

        all_live_models_path = MODELS_DIR / "all_live_models.joblib"
        joblib.dump(fitted_models, all_live_models_path)
        logger.info(f"Saved all fitted live models to {all_live_models_path}")

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_columns": feature_names,
        "test_set": (X_test, y_test),
        "all_models": fitted_models,
    }


# ========================================================================
# Prediction (inference-time) helpers
# ========================================================================

def load_live_model_metadata() -> dict:
    """
    Load the persisted live model comparison metadata JSON.

    Returns:
        The metadata dict written by `train_and_compare_live_models()`.

    Raises:
        LiveModelNotTrainedError: If no metadata file exists yet.
    """
    if not LIVE_MODEL_METADATA_PATH.exists():
        raise LiveModelNotTrainedError(
            f"No live model metadata found at {LIVE_MODEL_METADATA_PATH}. "
            f"Run `python -m src.models.live_win_probability` first."
        )
    with open(LIVE_MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_best_live_model():
    """Load the persisted best live model, raising `LiveModelNotTrainedError`
    with a helpful message if it hasn't been trained yet."""
    if not BEST_LIVE_MODEL_PATH.exists():
        raise LiveModelNotTrainedError(
            f"No trained live model found at {BEST_LIVE_MODEL_PATH}. "
            f"Run `python -m src.models.live_win_probability` first."
        )
    return joblib.load(BEST_LIVE_MODEL_PATH)


def _load_live_encoders() -> dict:
    """Load the persisted live-state LabelEncoders."""
    if not LIVE_ENCODERS_PATH.exists():
        raise LiveModelNotTrainedError(
            f"No live-state encoders found at {LIVE_ENCODERS_PATH}. "
            f"Run `python -m src.models.live_win_probability` first."
        )
    return joblib.load(LIVE_ENCODERS_PATH)


def predict_live_win_probability(
    batting_team: str,
    bowling_team: str,
    venue: str,
    target: int,
    runs_scored: int,
    wickets_lost: int,
    balls_bowled: int,
    batting_team_career_win_pct: float | None = None,
    bowling_team_career_win_pct: float | None = None,
) -> dict:
    """
    Predict the batting (chasing) team's win probability at a specific
    moment in a second-innings chase.

    Args:
        batting_team: Name of the team currently batting (chasing).
        bowling_team: Name of the team currently bowling (defending).
        venue: Venue name.
        target: Runs required to win (i.e. first innings score + 1).
        runs_scored: Runs scored by the batting team SO FAR (before the
            next ball).
        wickets_lost: Wickets lost by the batting team SO FAR.
        balls_bowled: Legal balls bowled in this innings SO FAR (0-119).
        batting_team_career_win_pct: Optional pre-computed recency-weighted
            career win pct for the batting team (see
            `src.features.engineering`). If None, defaults to a neutral
            0.5 -- for a quick estimate without recomputing full team
            history; pass the real value (e.g. from
            `src.features.engineering.compute_live_features`'s internals)
            for a more accurate prediction.
        bowling_team_career_win_pct: Same, for the bowling team.

    Returns:
        Dict with keys:
            "batting_team_win_probability": float in [0, 1]
            "bowling_team_win_probability": 1 - the above
            "predicted_winner": batting_team or bowling_team
            "model_name": name of the model used
            "runs_needed": target - runs_scored
            "balls_remaining": TOTAL_LEGAL_BALLS - balls_bowled
            "required_run_rate": runs needed per over, for display

    Raises:
        LiveModelNotTrainedError: If no trained live model exists yet.
        ValueError: If `wickets_lost` or `balls_bowled` are out of range.
    """
    if not (0 <= wickets_lost <= 10):
        raise ValueError(f"wickets_lost must be between 0 and 10, got {wickets_lost}.")
    if not (0 <= balls_bowled <= TOTAL_LEGAL_BALLS):
        raise ValueError(f"balls_bowled must be between 0 and {TOTAL_LEGAL_BALLS}, got {balls_bowled}.")

    model = _load_best_live_model()
    encoders = _load_live_encoders()
    metadata = load_live_model_metadata()

    runs_needed = max(target - runs_scored, 0)
    wickets_in_hand = max(10 - wickets_lost, 0)
    balls_remaining = max(TOTAL_LEGAL_BALLS - balls_bowled, 0)
    overs_bowled = balls_bowled / BALLS_PER_OVER
    overs_remaining = balls_remaining / BALLS_PER_OVER
    current_run_rate = (runs_scored / overs_bowled) if overs_bowled > 0 else 0.0
    required_run_rate = (runs_needed / overs_remaining) if overs_remaining > 0 else runs_needed * BALLS_PER_OVER

    batting_pct = 0.5 if batting_team_career_win_pct is None else batting_team_career_win_pct
    bowling_pct = 0.5 if bowling_team_career_win_pct is None else bowling_team_career_win_pct

    row = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "target": target,
        "runs_scored": runs_scored,
        "runs_needed": runs_needed,
        "wickets_lost": wickets_lost,
        "wickets_in_hand": wickets_in_hand,
        "balls_bowled": balls_bowled,
        "balls_remaining": balls_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
        "run_rate_diff": current_run_rate - required_run_rate,
        "batting_team_career_win_pct": batting_pct,
        "bowling_team_career_win_pct": bowling_pct,
    }
    single_row_df = pd.DataFrame([row])

    for col in ["batting_team", "bowling_team", "venue"]:
        le = encoders[col]
        known = set(le.classes_)
        unknown_label = len(le.classes_)
        value = str(single_row_df.at[0, col])
        single_row_df[f"{col}_enc"] = unknown_label if value not in known else int(le.transform([value])[0])

    X = single_row_df[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[0]
    # predict_proba's column order follows model.classes_, which for a
    # binary target fit as (0, 1) is [P(class 0), P(class 1)] -- class 1
    # means "batting team wins" (see live_state_engineering.TARGET_COLUMN).
    batting_win_prob = float(proba[1])

    return {
        "batting_team_win_probability": batting_win_prob,
        "bowling_team_win_probability": 1.0 - batting_win_prob,
        "predicted_winner": batting_team if batting_win_prob >= 0.5 else bowling_team,
        "model_name": metadata["best_model_name"],
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "required_run_rate": round(required_run_rate, 2),
    }


def compute_match_win_probability_timeline(match_id: str, matches_df: pd.DataFrame | None = None, deliveries_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """
    Reconstruct a single historical match's ball-by-ball state (via
    `src.features.live_state_engineering.get_match_state_timeline()`) and
    run the trained live model's `predict_proba` across every ball, for
    the "Live Win Probability" Streamlit page's match-replay chart
    (`src.visualization.plots.plot_win_probability_timeline`).

    Args:
        match_id: The match to replay.
        matches_df: Optional pre-loaded cleaned matches DataFrame (passed
            through to `get_match_state_timeline`).
        deliveries_df: Optional pre-loaded cleaned deliveries DataFrame.

    Returns:
        DataFrame with columns: `balls_bowled`, `over`, `ball`,
        `batting_team`, `bowling_team`, `is_wicket`,
        `batting_team_win_probability` -- or None if `match_id` isn't an
        eligible second-innings chase.

    Raises:
        LiveModelNotTrainedError: If no trained live model exists yet.
    """
    model = _load_best_live_model()
    encoders = _load_live_encoders()

    state_df = get_match_state_timeline(match_id, matches_df=matches_df, deliveries_df=deliveries_df)
    if state_df is None or state_df.empty:
        return None

    encoded = state_df.copy()
    for col in ["batting_team", "bowling_team", "venue"]:
        le = encoders[col]
        known = set(le.classes_)
        unknown_label = len(le.classes_)
        encoded[f"{col}_enc"] = encoded[col].astype(str).apply(
            lambda v: int(le.transform([v])[0]) if v in known else unknown_label
        )

    X = encoded[FEATURE_COLUMNS]
    probas = model.predict_proba(X)[:, 1]

    result = state_df[["balls_bowled", "over", "ball", "batting_team", "bowling_team", "is_wicket"]].copy()
    result["batting_team_win_probability"] = probas
    return result


if __name__ == "__main__":
    # Allows running `python -m src.models.live_win_probability` directly
    # to execute the full training + comparison pipeline and persist the
    # best model.
    run = train_and_compare_live_models()
    print(f"\nBest live model: {run['best_model_name']}")
    print("\nLive win-probability model comparison (cross-validated F1, grouped by match):")
    for name, metrics in sorted(
        run["results"].items(), key=lambda kv: kv[1]["cv_f1_mean"], reverse=True
    ):
        print(f"  {name:<22} CV F1={metrics['cv_f1_mean']:.3f}  Test Acc={metrics['test_accuracy']:.3f}  Test ROC-AUC={metrics['test_roc_auc']:.3f}")

    # Quick sanity-check prediction: a plausible mid-chase situation.
    demo = predict_live_win_probability(
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        venue="Wankhede Stadium",
        target=180,
        runs_scored=90,
        wickets_lost=3,
        balls_bowled=60,
    )
    print(f"\nDemo: Mumbai Indians chasing 180, 90/3 after 10 overs vs Chennai Super Kings")
    print(f"  Win probability: {demo['batting_team_win_probability']:.1%} (model: {demo['model_name']})")
