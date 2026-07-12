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
    - career win percentage prior to this match (RECENCY-WEIGHTED -- see
      the 2026-07-10 tuning note below)
    - recent-form win percentage (last 5 matches) prior to this match
    - venue win percentage prior to this match (this team, this venue --
      SHRUNK toward career win pct for small samples, see tuning note)
    - head-to-head win percentage prior to this match (this team vs this
      specific opponent, RECENCY-WEIGHTED)
    - number of career matches played prior to this match (lets the model
      learn to trust the above percentages more once they're based on a
      reasonable sample size, and lets cold-start rows -- where a team has
      no history -- be handled with a neutral 0.5 default)
    - team1-vs-team2 DIFFERENTIAL versions of the career/recent/venue win
      percentages (see tuning note)

------------------------------------------------------------------------
2026-07-10 FEATURE TUNING NOTE
------------------------------------------------------------------------
Three changes were made after reviewing initial model performance
(ROC-AUC ~0.53-0.61, barely above chance for some models), aimed at the
biggest identified source of noise: IPL squads turn over substantially
every season via trades/auctions, so a flat, all-time "career win %" mixes
in performance from a roster that may no longer resemble the current team.

    1. RECENCY WEIGHTING (`career_win_pct`, `h2h_team1_win_pct`): switched
       from a flat expanding mean over ALL prior matches to an
       exponentially decayed mean (`CAREER_FORM_HALFLIFE` / `H2H_HALFLIFE`
       matches), so recent matches (closer to the current squad) count
       more than matches from 10 seasons ago, without discarding long-run
       history entirely via a hard cutoff.
    2. SHRINKAGE (`venue_win_pct`): a team with 2 prior matches at a venue
       used to get a venue win% treated with the same confidence as a team
       with 40 -- now shrunk toward that team's career win pct, weighted
       by sample size (`VENUE_SHRINKAGE_K`), so small samples regress
       toward a more reliable prior instead of contributing noisy extremes
       (e.g. 1-for-1 = 100%).
    3. DIFFERENTIAL FEATURES (`diff_career_win_pct`, `diff_recent_win_pct`,
       `diff_venue_win_pct`): added explicit team1-minus-team2 versions of
       the three main win-pct features, so models (especially Logistic
       Regression) don't have to infer the difference between two
       separate columns themselves.

`compute_live_features()` mirrors all three changes for inference-time
consistency (see its docstring) -- training and prediction must compute
these statistics the same way, or the model would be scored on features
subtly different from what it learned on.

`city_enc` was also DROPPED from the feature set: it's almost entirely
redundant with `venue_enc` (each venue has one city) and was judged more
likely to add encoding noise than signal, especially given the modest
training-set size (~1,150 matches). `city` itself is retained on the
cleaned matches DataFrame for display/analytics purposes -- only the
model-input encoding was removed.

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

# --- 2026-07-10 tuning constants (see module docstring) ---
# Halflife, in matches, for the exponential decay applied to career win
# pct: a result from CAREER_FORM_HALFLIFE matches ago carries half the
# weight of the most recent match. ~30 matches is roughly one IPL season,
# so this keeps the current season's squad performance dominant while
# still drawing on multi-season history for teams with limited recent data.
CAREER_FORM_HALFLIFE: float = 30.0

# Halflife, in head-to-head MEETINGS (not matches), for head-to-head win
# pct decay. Meetings between two specific teams are much rarer than a
# team's overall matches (a few per season at most), so the halflife is
# expressed in a much smaller unit.
H2H_HALFLIFE: float = 6.0

# Shrinkage strength (in "equivalent prior matches") for venue win pct:
# a team's venue record is blended with its career win pct, weighted as
# if the career win pct were based on this many additional venue matches.
# Higher K = more shrinkage toward the career rate for small venue samples.
VENUE_SHRINKAGE_K: float = 5.0

# The final, ordered list of feature columns fed into every model. Centralized
# here so train.py, evaluate.py, and predict.py always agree on column order.
FEATURE_COLUMNS: list[str] = [
    "season",
    "match_month",
    "team1_enc",
    "team2_enc",
    "venue_enc",
    "toss_decision_enc",
    "toss_winner_is_team1",
    "team1_career_win_pct",
    "team2_career_win_pct",
    "diff_career_win_pct",
    "team1_recent_win_pct",
    "team2_recent_win_pct",
    "diff_recent_win_pct",
    "team1_venue_win_pct",
    "team2_venue_win_pct",
    "diff_venue_win_pct",
    "team1_matches_played",
    "team2_matches_played",
    "h2h_team1_win_pct",
    "h2h_matches_played",
]

TARGET_COLUMN: str = "team1_wins"

# Categorical columns that get LabelEncoder-encoded and persisted so the
# exact same encoding is reused at inference time in predict.py.
# NOTE: "city" was removed (2026-07-10 tuning) -- see module docstring.
CATEGORICAL_ENCODERS: list[str] = ["team1", "team2", "venue", "toss_decision"]


def _decayed_win_pct(is_winner: pd.Series, halflife: float) -> float:
    """
    Compute an exponentially recency-weighted win percentage over a
    chronologically-ordered (oldest-first) series of 1/0 win indicators,
    used at INFERENCE time (`compute_live_features()`) to mirror the
    decay-weighted statistics computed during training.

    The most recent entry gets weight 1.0; an entry `halflife` matches
    older gets weight 0.5; one `2 * halflife` matches older gets weight
    0.25; and so on. This is the same weighting scheme
    `pandas.Series.ewm(halflife=...)` applies internally, reimplemented
    here as a plain weighted average so it can be reused directly against
    an arbitrary already-materialized sub-series (e.g. a team's matches
    against one specific opponent) without going through a groupby.

    Args:
        is_winner: Series of 1/0 (or bool) win indicators, oldest first.
        halflife: Number of entries after which a result's weight halves.

    Returns:
        Weighted mean win rate (float), or `NEUTRAL_WIN_RATE` if the
        series is empty.
    """
    if is_winner.empty:
        return NEUTRAL_WIN_RATE
    n = len(is_winner)
    # Position 0 (oldest) gets the largest exponent (least weight);
    # position n-1 (most recent) gets exponent 0 (full weight 1.0).
    ages = np.arange(n - 1, -1, -1)
    weights = 0.5 ** (ages / halflife)
    return float(np.average(is_winner.astype(float), weights=weights))


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
    team match log: career win pct (recency-weighted), recent-form win
    pct, venue win pct (shrunk toward career win pct), and career matches
    played -- each using only matches strictly before the current row's
    match (via `.shift(1)` inside each team's group).

    Args:
        log: Output of `_build_team_match_log()`.

    Returns:
        The same long-format DataFrame with additional columns:
        `matches_played`, `career_win_pct`, `recent_win_pct`,
        `venue_matches_played`, `venue_win_pct`.
    """
    log = log.copy()

    grouped_team = log.groupby("team", group_keys=False)["is_winner"]

    # Career matches played prior to this match = count of PRIOR rows for
    # this team (cumcount is naturally "prior count", no shift needed).
    log["matches_played"] = grouped_team.cumcount()

    # Career win pct prior to this match = EXPONENTIALLY DECAYED expanding
    # mean of all PRIOR results for this team (2026-07-10 tuning: was a
    # flat expanding mean; see module docstring). shift(1) excludes the
    # current row; ewm(halflife=...) down-weights older results so a
    # team's current-season roster dominates its own "career" rate.
    log["career_win_pct"] = (
        grouped_team.apply(
            lambda s: s.shift(1).ewm(halflife=CAREER_FORM_HALFLIFE, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    )
    log["career_win_pct"] = log["career_win_pct"].fillna(NEUTRAL_WIN_RATE)

    # Recent-form win pct = rolling mean of the last RECENT_FORM_WINDOW
    # prior matches for this team -- a short, unweighted window capturing
    # "hot/cold streak" form, distinct from the longer-run career rate.
    log["recent_win_pct"] = (
        log.groupby("team", group_keys=False)["is_winner"]
        .apply(lambda s: s.shift(1).rolling(window=RECENT_FORM_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    log["recent_win_pct"] = log["recent_win_pct"].fillna(NEUTRAL_WIN_RATE)

    # Venue win pct prior to this match: raw prior wins/matches at this
    # specific (team, venue), then SHRUNK toward this team's (already
    # decay-weighted) career_win_pct, weighted by how many prior matches
    # the team has actually played at this venue (2026-07-10 tuning --
    # see module docstring). A team with 0 prior matches at a venue gets
    # exactly its career_win_pct; a team with many prior matches there is
    # increasingly dominated by its actual venue record.
    log["venue_matches_played"] = log.groupby(["team", "venue"], group_keys=False).cumcount()
    venue_wins_prior = (
        log.groupby(["team", "venue"], group_keys=False)["is_winner"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    log["venue_win_pct"] = (
        venue_wins_prior + VENUE_SHRINKAGE_K * log["career_win_pct"]
    ) / (log["venue_matches_played"] + VENUE_SHRINKAGE_K)

    return log


def _add_head_to_head_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for every match, team1's historical win percentage against
    THIS SPECIFIC opponent (team2) prior to this match, plus the number of
    prior head-to-head meetings.

    2026-07-10 tuning: `h2h_team1_win_pct` is now EXPONENTIALLY
    RECENCY-WEIGHTED (halflife = `H2H_HALFLIFE` meetings, not matches --
    see module docstring) rather than a flat average over the full
    head-to-head history, for the same squad-turnover reasoning as
    `career_win_pct`. `h2h_matches_played` remains a plain (undecayed)
    count, since it's used by the model as a sample-size signal, not a
    rate.

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

    # Exponential decay factor applied per meeting: a pair's accumulated
    # weighted stats are multiplied by ALPHA every time a new meeting is
    # recorded, which reproduces halflife-based decay incrementally
    # (equivalent to `pandas.ewm(halflife=H2H_HALFLIFE)` but computed
    # match-by-match, since -- unlike `career_win_pct` -- which side is
    # "team1" can flip between two teams' meetings, so a single per-pair
    # scalar can't be reused the way a per-team column can).
    alpha = 0.5 ** (1.0 / H2H_HALFLIFE)

    # pair_stats[frozenset({teamA, teamB})] -> {
    #     "weighted_wins": {team_name: decayed win count, ...},
    #     "weighted_matches": decayed total meeting count,
    #     "raw_count": true (undecayed) meeting count,
    # }
    pair_stats: dict[frozenset, dict] = {}

    for i, row in enumerate(df.itertuples(index=False)):
        pair_key = frozenset({row.team1, row.team2})
        stats = pair_stats.get(pair_key)

        if stats is not None and stats["weighted_matches"] > 0:
            team1_weighted_wins = stats["weighted_wins"].get(row.team1, 0.0)
            h2h_win_pct[i] = team1_weighted_wins / stats["weighted_matches"]
        if stats is not None:
            h2h_matches[i] = stats["raw_count"]

        if stats is None:
            stats = {"weighted_wins": {}, "weighted_matches": 0.0, "raw_count": 0}

        # Decay existing accumulated weight, then add this match's result.
        stats["weighted_wins"] = {team: w * alpha for team, w in stats["weighted_wins"].items()}
        stats["weighted_matches"] *= alpha
        stats["weighted_wins"][row.winner] = stats["weighted_wins"].get(row.winner, 0.0) + 1.0
        stats["weighted_matches"] += 1.0
        stats["raw_count"] += 1

        pair_stats[pair_key] = stats

    df["h2h_team1_win_pct"] = h2h_win_pct
    df["h2h_matches_played"] = h2h_matches
    return df


def _encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns (team1, team2, venue, toss_decision)
    into integer columns suffixed `_enc`.

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

    # --- Differential features (2026-07-10 tuning; see module docstring) ---
    df["diff_career_win_pct"] = df["team1_career_win_pct"] - df["team2_career_win_pct"]
    df["diff_recent_win_pct"] = df["team1_recent_win_pct"] - df["team2_recent_win_pct"]
    df["diff_venue_win_pct"] = df["team1_venue_win_pct"] - df["team2_venue_win_pct"]

    # --- Simple derived / pre-match-knowable columns ---
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    # NOTE: `is_day_night` was dropped as a feature -- the new dataset has
    # no day/night or match start-time field anywhere (see the migration
    # note in src/utils/config.py). No replacement was substituted since
    # there's no comparably meaningful pre-match-knowable alternative
    # available in this dataset; the model simply has one fewer feature.
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

    2026-07-10 tuning: this function mirrors the three changes made in
    `_add_rolling_team_stats()` / `_add_head_to_head_stats()` (recency
    weighting for career/h2h win pct, venue shrinkage, differential
    features) -- see the module docstring. Training and inference MUST
    compute these statistics the same way, or the model would be scored
    on features subtly different from what it learned on. Concretely:
        - `_team_snapshot()`'s career win pct now uses
          `.ewm(halflife=CAREER_FORM_HALFLIFE).mean()` instead of a flat
          `.mean()` (equivalent to the training-time decayed expanding
          mean, evaluated at the most recent point -- i.e. "as of now").
        - `_venue_snapshot()` now shrinks toward the team's career win pct
          using `VENUE_SHRINKAGE_K`, matching the training-time formula
          exactly.
        - `_h2h_snapshot()` now uses `_decayed_win_pct()` (halflife =
          `H2H_HALFLIFE` meetings) instead of a flat `.mean()`.
        - `diff_career_win_pct` / `diff_recent_win_pct` /
          `diff_venue_win_pct` are computed and included in the output row.
        - `city_enc` is no longer produced (city was dropped from
          `CATEGORICAL_ENCODERS`); the `city` argument is still accepted
          for API compatibility and potential future/display use, but no
          longer feeds into the model.

    Args:
        team1: Name of the first team (post `TEAM_NAME_MAPPING` normalization).
        team2: Name of the second team.
        venue: Venue name.
        city: City name (accepted for API compatibility; not currently
            used as a model feature -- see tuning note above).
        toss_winner: Name of the team that won the toss (must be team1 or team2).
        toss_decision: "bat" or "field".
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
        """Recency-weighted career win pct, recent win pct, and matches
        played for `team`, using ALL history available (i.e. as of "now").
        The career figure uses the same ewm-decay weighting as training
        (see 2026-07-10 tuning note above)."""
        team_log = log[log["team"] == team]
        if team_log.empty:
            return NEUTRAL_WIN_RATE, NEUTRAL_WIN_RATE, 0
        career = team_log["is_winner"].ewm(halflife=CAREER_FORM_HALFLIFE, min_periods=1).mean().iloc[-1]
        recent = team_log["is_winner"].tail(RECENT_FORM_WINDOW).mean()
        played = len(team_log)
        return float(career), float(recent), int(played)

    def _venue_snapshot(team: str, venue_name: str, career_pct: float) -> float:
        """Venue win pct for `team` at `venue_name`, shrunk toward
        `career_pct` by sample size -- matches the training-time shrinkage
        formula exactly (see 2026-07-10 tuning note above)."""
        venue_log = log[(log["team"] == team) & (log["venue"] == venue_name)]
        n = len(venue_log)
        wins = float(venue_log["is_winner"].sum())
        return (wins + VENUE_SHRINKAGE_K * career_pct) / (n + VENUE_SHRINKAGE_K)

    def _h2h_snapshot(team_a: str, team_b: str) -> tuple[float, int]:
        """Recency-weighted head-to-head win pct for `team_a` against
        `team_b`, using the same decay halflife as training."""
        h2h_log = log[(log["team"] == team_a) & (log["opponent"] == team_b)]
        if h2h_log.empty:
            return NEUTRAL_WIN_RATE, 0
        pct = _decayed_win_pct(h2h_log["is_winner"], H2H_HALFLIFE)
        return pct, int(len(h2h_log))

    team1_career, team1_recent, team1_played = _team_snapshot(team1)
    team2_career, team2_recent, team2_played = _team_snapshot(team2)
    team1_venue = _venue_snapshot(team1, venue, team1_career)
    team2_venue = _venue_snapshot(team2, venue, team2_career)
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
        "team1_career_win_pct": team1_career,
        "team2_career_win_pct": team2_career,
        "diff_career_win_pct": team1_career - team2_career,
        "team1_recent_win_pct": team1_recent,
        "team2_recent_win_pct": team2_recent,
        "diff_recent_win_pct": team1_recent - team2_recent,
        "team1_venue_win_pct": team1_venue,
        "team2_venue_win_pct": team2_venue,
        "diff_venue_win_pct": team1_venue - team2_venue,
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
