"""
predict.py
==========
Inference layer for the Cricket Analytics & Match Prediction Platform.

This module loads the persisted best model (trained by
`src/models/train.py`) and the fitted feature encoders (saved by
`src/features/engineering.py`), and exposes a single high-level function,
`predict_match()`, that takes plain human inputs (team names, venue, toss)
and returns a winner prediction with a probability/confidence score.

This is the module the Streamlit "Match Prediction" page calls directly --
it deliberately knows nothing about Streamlit, so it's equally usable from
a script, a notebook, or a future API layer.

Usage:
    from src.models.predict import predict_match

    result = predict_match(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        city="Mumbai",
        toss_winner="Mumbai Indians",
        toss_decision="bat",
    )
    print(result["predicted_winner"], result["win_probability"])
"""

from __future__ import annotations

import functools
from datetime import datetime

import joblib
import pandas as pd

from src.data.preprocess import preprocess_all
from src.features.engineering import compute_live_features
from src.utils.config import BEST_MODEL_PATH, ENCODERS_PATH, MODEL_METADATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotTrainedError(Exception):
    """
    Raised when `predict_match()` (or any loader in this module) is called
    before a model has been trained and persisted.

    Using a dedicated exception (rather than a generic FileNotFoundError)
    lets the Streamlit app catch this specific case and show a friendly
    "please run the training pipeline first" message instead of a raw
    traceback.
    """
    pass


@functools.lru_cache(maxsize=1)
def load_best_model():
    """
    Load the persisted best model from `BEST_MODEL_PATH`, cached after the
    first call so repeated predictions (e.g. many Streamlit interactions
    in one session) don't re-read the file from disk every time.

    Returns:
        The fitted best estimator (whatever model `train.py` selected).

    Raises:
        ModelNotTrainedError: If no trained model exists yet at
            `BEST_MODEL_PATH`.
    """
    if not BEST_MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {BEST_MODEL_PATH}. Run "
            f"`python -m src.models.train` first to train and save a model."
        )
    logger.info(f"Loading best model from {BEST_MODEL_PATH}")
    return joblib.load(BEST_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_encoders() -> dict:
    """
    Load the persisted LabelEncoders dict from `ENCODERS_PATH`, cached
    after the first call for the same reason as `load_best_model()`.

    Returns:
        Dict mapping categorical column name -> fitted LabelEncoder.

    Raises:
        ModelNotTrainedError: If no encoders file exists yet (i.e. the
            feature engineering / training pipeline hasn't been run).
    """
    if not ENCODERS_PATH.exists():
        raise ModelNotTrainedError(
            f"No fitted encoders found at {ENCODERS_PATH}. Run "
            f"`python -m src.models.train` (or `python -m src.features.engineering`) "
            f"first."
        )
    logger.info(f"Loading fitted encoders from {ENCODERS_PATH}")
    return joblib.load(ENCODERS_PATH)


@functools.lru_cache(maxsize=1)
def load_model_metadata() -> dict:
    """
    Load the JSON metadata written by `train.py` describing which model
    was selected, its metrics, and the feature columns used.

    Returns:
        Dict parsed from `MODEL_METADATA_PATH`.

    Raises:
        ModelNotTrainedError: If no metadata file exists yet.
    """
    if not MODEL_METADATA_PATH.exists():
        raise ModelNotTrainedError(
            f"No model metadata found at {MODEL_METADATA_PATH}. Run "
            f"`python -m src.models.train` first."
        )
    import json

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_matches_for_inference() -> pd.DataFrame:
    """
    Load and clean the matches dataset once, cached for the lifetime of
    the process, so every call to `predict_match()` doesn't re-run the
    full preprocessing pipeline from raw CSVs (which would be slow inside
    an interactive Streamlit session with many predictions per minute).

    Returns:
        Cleaned matches DataFrame (output of `preprocess_all()["matches"]`).
    """
    cleaned = preprocess_all(save=False)
    return cleaned["matches"]


def get_known_teams() -> list[str]:
    """
    Return the sorted list of team names the model was trained on (i.e.
    valid values for `team1`/`team2`/`toss_winner` in `predict_match()`),
    for populating dropdowns in the Streamlit UI.

    Returns:
        Sorted list of team name strings.
    """
    matches_df = _load_matches_for_inference()
    teams = sorted(set(matches_df["team1"]) | set(matches_df["team2"]))
    return teams


def get_known_venues() -> pd.DataFrame:
    """
    Return the distinct (venue, city) pairs seen in the historical data,
    for populating dependent venue/city dropdowns in the Streamlit UI.

    Returns:
        DataFrame with columns "venue" and "city", one row per distinct
        venue, sorted alphabetically by venue.
    """
    matches_df = _load_matches_for_inference()
    venues = matches_df[["venue", "city"]].drop_duplicates().sort_values("venue").reset_index(drop=True)
    return venues


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    city: str,
    toss_winner: str,
    toss_decision: str,
    season: int | None = None,
    match_month: int | None = None,
) -> dict:
    """
    Predict the winner of a hypothetical match between `team1` and `team2`.

    Args:
        team1: First team's name (must be a team name the model has seen
            historically -- see `get_known_teams()`).
        team2: Second team's name.
        venue: Venue name (see `get_known_venues()`; unseen venues are
            handled gracefully via the encoder's "unknown" bucket, but
            with reduced confidence since there's no venue-specific history).
        city: City the venue is in.
        toss_winner: Name of the team that won the toss (must equal
            `team1` or `team2`).
        toss_decision: "bat" or "field" -- what the toss winner chose to do.
        season: Season year to associate with the hypothetical match.
            Defaults to the current calendar year if not given.
        match_month: Calendar month (1-12). Defaults to the current month
            if not given.

    Returns:
        Dict with keys:
            "predicted_winner": team name predicted to win.
            "win_probability": probability (0-1) assigned to the predicted
                winner.
            "team1_win_probability": probability (0-1) that team1 wins.
            "team2_win_probability": probability (0-1) that team2 wins.
            "model_name": name of the model used (from metadata).
            "features_used": the single-row feature DataFrame, useful for
                debugging or displaying "why" in the UI.

    Raises:
        ModelNotTrainedError: If the model/encoders haven't been trained yet.
        ValueError: If `toss_winner` is not `team1` or `team2`, or if
            `team1 == team2`.

    Example:
        >>> result = predict_match(
        ...     team1="Mumbai Indians", team2="Chennai Super Kings",
        ...     venue="Wankhede Stadium", city="Mumbai",
        ...     toss_winner="Mumbai Indians", toss_decision="bat",
        ... )
        >>> result["predicted_winner"] in {"Mumbai Indians", "Chennai Super Kings"}
        True
    """
    if team1 == team2:
        raise ValueError("team1 and team2 must be different teams.")

    model = load_best_model()
    encoders = load_encoders()
    metadata = load_model_metadata()
    matches_df = _load_matches_for_inference()

    now = datetime.now()
    season = season if season is not None else now.year
    match_month = match_month if match_month is not None else now.month

    features = compute_live_features(
        team1=team1,
        team2=team2,
        venue=venue,
        city=city,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        season=season,
        match_month=match_month,
        encoders=encoders,
        matches_df=matches_df,
    )

    proba = model.predict_proba(features)[0]
    team1_win_probability = float(proba[1])
    team2_win_probability = float(proba[0])

    predicted_winner = team1 if team1_win_probability >= team2_win_probability else team2
    win_probability = max(team1_win_probability, team2_win_probability)

    logger.info(
        f"Predicted {team1} vs {team2} @ {venue}: {predicted_winner} "
        f"({win_probability:.1%}) using {metadata['best_model_name']}"
    )

    return {
        "predicted_winner": predicted_winner,
        "win_probability": win_probability,
        "team1_win_probability": team1_win_probability,
        "team2_win_probability": team2_win_probability,
        "model_name": metadata["best_model_name"],
        "features_used": features,
    }


if __name__ == "__main__":
    # Allows running `python -m src.models.predict` directly as a quick
    # sanity check / demo of a single prediction.
    demo_teams = get_known_teams()
    team_a, team_b = demo_teams[0], demo_teams[1]
    venues = get_known_venues()
    demo_venue, demo_city = venues.iloc[0]["venue"], venues.iloc[0]["city"]

    result = predict_match(
        team1=team_a,
        team2=team_b,
        venue=demo_venue,
        city=demo_city,
        toss_winner=team_a,
        toss_decision="bat",
    )
    print(f"{team_a} vs {team_b} @ {demo_venue}")
    print(f"Predicted winner: {result['predicted_winner']} ({result['win_probability']:.1%})")
    print(f"Model used: {result['model_name']}")
