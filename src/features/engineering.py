"""
engineering.py
===============
Feature engineering layer for the Cricket Analytics & Match Prediction
Platform.

This module turns the *cleaned* matches DataFrame produced by
`src/data/preprocess.py` into a model-ready feature matrix for predicting
match winners.

Design principle -- NO LEAKAGE:
    Every engineered feature describing "how good is this team" must only
    use information available *before* the match was played. Concretely:
    we sort matches chronologically and use `.shift(1)` / expanding or
    rolling windows on the *prior* matches of a team, never including the
    current match's own result. Columns describing the outcome of the
    CURRENT match (scores, wickets, overs, margins) are never used as
    features, since none of that is known before the match is played --
    only `team1`, `team2`, `venue`, `city`, `toss_winner`, `toss_decision`,
    `season`, `date`, and `stage` are "pre-match knowable" raw inputs.

Engineered features (all leakage-safe, all relative to a single team
appearing as either team1 or team2 in a given match):
    - career win percentage prior to this match
    - recent-form win percentage (last 5 matches) prior to this match
    - venue win percentage prior to this match (this team, this venue)
    - head-to-head win percentage prior to this match (this team vs this
      specific opponent)
    - number of career matches played prior to this match (lets the model
      learn to trust the above percentages more once they're based on a
      reasonable sample size, and lets cold-start rows -- where a team has
      no history -- be handled with a neutral 0.5 default)

Usage:
    from src.features.engineering import build_model_dataset

    X, y, feature_names, encoders = build_model_dataset()

For live / hypothetical match prediction (no result to learn from), see
`compute_live_features()`, which derives the same statistics as of "today"
(or as of a given date) for an arbitrary team1/team2/venue/toss matchup.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.preprocess import preprocess_all
from src.utils.config import (
    ENCODERS_PATH,
    MODEL_FEATURES_PATH,
    ensure_directories_exist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Recent-form window size (number of trailing matches used for "form").
RECENT_FORM_WINDOW: int = 5

# Neutral default used when a team/venue/match-up has no prior history yet
# (e.g. a brand-new franchise's first ever match). 0.5 represents "unknown,
# assume even odds" rather than biasing the model toward 0 or 1.
NEUTRAL_WIN_RATE: float = 0.5

# The final, ordered list of feature columns fed into every model. Centralized
# here so train.py, evaluate.py, and predict.py always agree on column order.
FEATURE_COLUMNS: list[str] = [
    "season",
    "match_month",
    "team1_enc",
    "team2_enc",
    "venue_enc",
    "city_enc",
    "toss_decision_enc",
    "toss_winner_is_team1",
    "is_day_night",
    "team1_career_win_pct",
    "team2_career_win_pct",
    "team1_recent_win_pct",
    "team2_recent_win_pct",
    "team1_venue_win_pct",
    "team2_venue_win_pct",
    "team1_matches_played",
    "team2_matches_played",
    "h2h_team1_win_pct",
    "h2h_matches_played",
]

TARGET_COLUMN: str = "team1_wins"

# Categorical columns that get LabelEncoder-encoded and persisted so the
# exact same encoding is reused at inference time in predict.py.
CATEGORICAL_ENCODERS: list[str] = ["team1", "team2", "venue", "city", "toss_decision"]


def _build_team_match_log(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape match-level data (one row per match) into team-perspective
    "long" format (two rows per match: one for each participating team),
    sorted chronologically. This long format is the basis for every
    rolling/expanding team-level statistic computed below.

    Args:
        matches_df: Cleaned matches DataFrame (must include `is_no_result`).

    Returns:
        DataFrame with one row per (match, team) pair, columns:
            match_id, date, season, team, opponent, venue, is_winner
        Sorted by date ascending, then match_id, then team-slot, so that
        `.shift(1)` / rolling windows computed per-team reflect strictly
        prior matches.
    """
    base = matches_df[~matches_df["is_no_result"]].copy()

    team1_rows = base[["match_id", "date", "season", "team1", "team2", "venue", "winner"]].rename(
        columns={"team1": "team", "team2": "opponent"}
    )
    team1_rows["is_winner"] = (team1_rows["team"] == team1_rows["winner"]).astype(int)

    team2_rows = base[["match_id", "date", "season", "team2", "team1", "venue", "winner"]].rename(
        columns={"team2": "team", "team1": "opponent"}
    )
    team2_rows["is_winner"] = (team2_rows["team"] == team2_rows["winner"]).astype(int)

    log = pd.concat([team1_rows, team2_rows], ignore_index=True)
    log = log.drop(columns=["winner"])
    log["date"] = pd.to_datetime(log["date"])
    log = log.sort_values(["date", "match_id"]).reset_index(drop=True)
    return log


def _add_rolling_team_stats(log: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe, per-team rolling statistics on the long-format
    team match log: career win pct, recent-form win pct, venue win pct,
    and career matches played -- each using only matches strictly before
    the current row's match (via `.shift(1)` inside each team's group).

    Args:
        log: Output of `_build_team_match_log()`.

    Returns:
        The same long-format DataFrame with four additional columns:
        `career_win_pct`, `recent_win_pct`, `venue_win_pct`, `matches_played`.
    """
    log = log.copy()

    grouped_team = log.groupby("team", group_keys=False)["is_winner"]

    # Career win pct prior to this match = expanding mean of all PRIOR
    # results for this team. shift(1) excludes the current row.
    log["matches_played"] = grouped_team.cumcount()
    log["career_win_pct"] = (
        grouped_team.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )

    # Recent-form win pct = rolling mean of the last RECENT_FORM_WINDOW
    # prior matches for this team.
    log["recent_win_pct"] = (
        log.groupby("team", group_keys=False)["is_winner"]
        .apply(lambda s: s.shift(1).rolling(window=RECENT_FORM_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Venue win pct prior to this match = expanding mean of this team's
    # prior results AT THIS SPECIFIC VENUE.
    log["venue_win_pct"] = (
        log.groupby(["team", "venue"], group_keys=False)["is_winner"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Cold-start handling: no prior history yet -> neutral 0.5.
    for col in ("career_win_pct", "recent_win_pct", "venue_win_pct"):
        log[col] = log[col].fillna(NEUTRAL_WIN_RATE)

    return log


def _add_head_to_head_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for every match, team1's historical win percentage against
    THIS SPECIFIC opponent (team2) prior to this match, plus the number of
    prior head-to-head meetings.

    Args:
        matches_df: Cleaned matches DataFrame, sorted chronologically.

    Returns:
        DataFrame indexed identically to `matches_df` (after sorting by
        date) with two new columns: `h2h_team1_win_pct`, `h2h_matches_played`.
    """
    df = matches_df[~matches_df["is_no_result"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    h2h_win_pct = np.full(len(df), NEUTRAL_WIN_RATE, dtype=float)
    h2h_matches = np.zeros(len(df), dtype=int)

    # pair_history[frozenset({teamA, teamB})] -> list of winners seen so far
    pair_history: dict[frozenset, list[str]] = {}

    for i, row in enumerate(df.itertuples(index=False)):
        pair_key = frozenset({row.team1, row.team2})
        history = pair_history.get(pair_key, [])

        h2h_matches[i] = len(history)
        if history:
            team1_wins_in_pair = sum(1 for w in history if w == row.team1)
            h2h_win_pct[i] = team1_wins_in_pair / len(history)

        history.append(row.winner)
        pair_history[pair_key] = history

    df["h2h_team1_win_pct"] = h2h_win_pct
    df["h2h_matches_played"] = h2h_matches
    return df


def _encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns (team1, team2, venue, city,
    toss_decision) into integer columns suffixed `_enc`.

    Args:
        df: DataFrame containing the raw categorical columns.
        fit: If True, fit new LabelEncoders on `df` and return them. If
            False, reuse the encoders passed in via `encoders` (used at
            inference time so categories map to the same integers the
            model was trained on).
        encoders: Required when `fit=False`. Dict mapping column name ->
            fitted LabelEncoder.

    Returns:
        Tuple of (DataFrame with new `_enc` columns added, dict of
        LabelEncoders used).
    """
    df = df.copy()
    encoders = {} if encoders is None else encoders

    for col in CATEGORICAL_ENCODERS:
        if fit:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Unseen categories at inference time (e.g. a brand-new venue)
            # are mapped to a new "unknown" bucket appended after the
            # encoder's known classes, rather than raising KeyError.
            known = set(le.classes_)
            unknown_label = len(le.classes_)
            df[f"{col}_enc"] = df[col].astype(str).apply(
                lambda v: int(le.transform([v])[0]) if v in known else unknown_label
            )

    return df, encoders


def build_model_dataset(save: bool = True) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """
    Run the full feature engineering pipeline end-to-end: load + clean raw
    data, compute leakage-safe rolling team statistics, encode categorical
    columns, and assemble the final model-ready feature matrix.

    No-result/tied matches (`is_no_result == True`) are excluded entirely,
    since there is no valid winner to learn from.

    Args:
        save: If True (default), persist the feature matrix to
            `MODEL_FEATURES_PATH` and the fitted LabelEncoders to
            `ENCODERS_PATH` via joblib, so `src/models/train.py` and
            `src/models/predict.py` can reuse them without recomputing.

    Returns:
        Tuple of:
            X: Feature matrix (DataFrame), columns = FEATURE_COLUMNS.
            y: Target Series (1 if team1 won, 0 if team2 won).
            feature_names: FEATURE_COLUMNS (convenience passthrough).
            encoders: Dict of fitted LabelEncoders, keyed by column name.

    Example:
        >>> X, y, feature_names, encoders = build_model_dataset()
        >>> X.shape[0] == y.shape[0]
        True
    """
    logger.info("Building model-ready feature dataset...")
    ensure_directories_exist()

    cleaned = preprocess_all(save=False)
    matches_df = cleaned["matches"]

    # --- Per-team rolling stats (career / recent-form / venue) ---
    log = _build_team_match_log(matches_df)
    log = _add_rolling_team_stats(log)

    # Split the long-format log back into team1-perspective and
    # team2-perspective slices and merge each onto the match table.
    team1_stats = log.rename(
        columns={
            "team": "team1",
            "career_win_pct": "team1_career_win_pct",
            "recent_win_pct": "team1_recent_win_pct",
            "venue_win_pct": "team1_venue_win_pct",
            "matches_played": "team1_matches_played",
        }
    )[["match_id", "team1", "team1_career_win_pct", "team1_recent_win_pct",
       "team1_venue_win_pct", "team1_matches_played"]]

    team2_stats = log.rename(
        columns={
            "team": "team2",
            "career_win_pct": "team2_career_win_pct",
            "recent_win_pct": "team2_recent_win_pct",
            "venue_win_pct": "team2_venue_win_pct",
            "matches_played": "team2_matches_played",
        }
    )[["match_id", "team2", "team2_career_win_pct", "team2_recent_win_pct",
       "team2_venue_win_pct", "team2_matches_played"]]

    df = matches_df[~matches_df["is_no_result"]].copy()
    df = df.merge(team1_stats, on=["match_id", "team1"], how="left")
    df = df.merge(team2_stats, on=["match_id", "team2"], how="left")

    # --- Head-to-head stats ---
    h2h = _add_head_to_head_stats(matches_df)[
        ["match_id", "h2h_team1_win_pct", "h2h_matches_played"]
    ]
    df = df.merge(h2h, on="match_id", how="left")

    # --- Simple derived / pre-match-knowable columns ---
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["is_day_night"] = df["is_day_night"].astype(int)
    df[TARGET_COLUMN] = (df["winner"] == df["team1"]).astype(int)

    # --- Categorical encoding ---
    df, encoders = _encode_categoricals(df, fit=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    logger.info(
        f"Feature dataset built: {X.shape[0]:,} matches x {X.shape[1]} features "
        f"(target balance: {y.mean():.1%} team1 wins)."
    )

    if save:
        model_features_df = X.copy()
        model_features_df[TARGET_COLUMN] = y
        model_features_df.to_csv(MODEL_FEATURES_PATH, index=False)
        joblib.dump(encoders, ENCODERS_PATH)
        logger.info(f"Saved feature matrix to {MODEL_FEATURES_PATH}")
        logger.info(f"Saved fitted encoders to {ENCODERS_PATH}")

    return X, y, FEATURE_COLUMNS, encoders


def compute_live_features(
    team1: str,
    team2: str,
    venue: str,
    city: str,
    toss_winner: str,
    toss_decision: str,
    is_day_night: bool,
    season: int,
    match_month: int,
    encoders: dict,
    matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute the same leakage-safe feature set as `build_model_dataset()`,
    but for a single HYPOTHETICAL future match (no result to learn from
    yet) -- used by `src/models/predict.py` and the Streamlit "Match
    Prediction" page to build a feature vector at inference time.

    All rolling statistics (career/recent/venue/head-to-head win
    percentages) are computed using the ENTIRE historical match record
    available (i.e. "as of today"), since a hypothetical match is assumed
    to occur after every match already in the dataset.

    Args:
        team1: Name of the first team (post `TEAM_NAME_MAPPING` normalization).
        team2: Name of the second team.
        venue: Venue name.
        city: City name.
        toss_winner: Name of the team that won the toss (must be team1 or team2).
        toss_decision: "bat" or "field".
        is_day_night: Whether this is a day/night match.
        season: Season year to associate with the hypothetical match.
        match_month: Calendar month (1-12) to associate with the hypothetical match.
        encoders: Fitted LabelEncoders dict, as returned by `build_model_dataset()`
            (load via `joblib.load(ENCODERS_PATH)` if calling outside that
            same process).
        matches_df: Optional pre-loaded, pre-cleaned matches DataFrame. If
            None, it is recomputed via `preprocess_all()` (slower, but
            convenient for one-off calls).

    Returns:
        A single-row DataFrame with columns = FEATURE_COLUMNS, ready to be
        passed directly into a trained model's `.predict()` / `.predict_proba()`.

    Raises:
        ValueError: If `toss_winner` is not one of `team1` or `team2`.
    """
    if toss_winner not in (team1, team2):
        raise ValueError(
            f"toss_winner ({toss_winner!r}) must be either team1 ({team1!r}) "
            f"or team2 ({team2!r})."
        )

    if matches_df is None:
        cleaned = preprocess_all(save=False)
        matches_df = cleaned["matches"]

    log = _build_team_match_log(matches_df)

    def _team_snapshot(team: str) -> tuple[float, float, int]:
        """Career win pct, recent win pct, and matches played for `team`,
        using ALL history available (i.e. as of "now")."""
        team_log = log[log["team"] == team]
        if team_log.empty:
            return NEUTRAL_WIN_RATE, NEUTRAL_WIN_RATE, 0
        career = team_log["is_winner"].mean()
        recent = team_log["is_winner"].tail(RECENT_FORM_WINDOW).mean()
        played = len(team_log)
        return float(career), float(recent), int(played)

    def _venue_snapshot(team: str, venue_name: str) -> float:
        venue_log = log[(log["team"] == team) & (log["venue"] == venue_name)]
        if venue_log.empty:
            return NEUTRAL_WIN_RATE
        return float(venue_log["is_winner"].mean())

    def _h2h_snapshot(team_a: str, team_b: str) -> tuple[float, int]:
        h2h_log = log[(log["team"] == team_a) & (log["opponent"] == team_b)]
        if h2h_log.empty:
            return NEUTRAL_WIN_RATE, 0
        return float(h2h_log["is_winner"].mean()), int(len(h2h_log))

    team1_career, team1_recent, team1_played = _team_snapshot(team1)
    team2_career, team2_recent, team2_played = _team_snapshot(team2)
    team1_venue = _venue_snapshot(team1, venue)
    team2_venue = _venue_snapshot(team2, venue)
    h2h_win_pct, h2h_played = _h2h_snapshot(team1, team2)

    row = {
        "season": season,
        "match_month": match_month,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_decision": toss_decision,
        "toss_winner_is_team1": int(toss_winner == team1),
        "is_day_night": int(is_day_night),
        "team1_career_win_pct": team1_career,
        "team2_career_win_pct": team2_career,
        "team1_recent_win_pct": team1_recent,
        "team2_recent_win_pct": team2_recent,
        "team1_venue_win_pct": team1_venue,
        "team2_venue_win_pct": team2_venue,
        "team1_matches_played": team1_played,
        "team2_matches_played": team2_played,
        "h2h_team1_win_pct": h2h_win_pct,
        "h2h_matches_played": h2h_played,
    }
    single_row_df = pd.DataFrame([row])
    encoded_df, _ = _encode_categoricals(single_row_df, fit=False, encoders=encoders)

    return encoded_df[FEATURE_COLUMNS]


if __name__ == "__main__":
    # Allows running `python -m src.features.engineering` directly as a
    # quick sanity check that the full feature pipeline runs end-to-end.
    X, y, feature_names, encoders = build_model_dataset()
    print(f"X shape: {X.shape}")
    print(f"y balance: {y.mean():.3f}")
    print(X.head())
