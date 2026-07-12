"""
live_state_engineering.py
==========================
Feature engineering for the IN-MATCH ("live") win-probability model --
the counterpart to `src/features/engineering.py`'s PRE-MATCH model.

------------------------------------------------------------------------
WHY THIS MODULE EXISTS (see docs/MODEL_CARD.md "Part 2" for full detail)
------------------------------------------------------------------------
The pre-match model (`src/features/engineering.py`) predicts a winner
using only information known before a ball is bowled -- team identity,
venue, toss, historical form. Its documented, honest performance is close
to chance (test ROC-AUC ~0.5), because T20 cricket's outcome is dominated
by what happens *during* the match, which pre-match features structurally
cannot see.

This module instead builds a dataset where each row is a single BALL in
the second innings of a chase, with features describing the match state
at that exact moment (target, runs needed, wickets in hand, balls
remaining, run rates) -- i.e. it reframes the problem from "who will win
this match" (hard, low ceiling) to "given the game state right now, who's
ahead" (well-posed, high ceiling), which is the same problem broadcast
win-probability graphics solve.

------------------------------------------------------------------------
SCOPE -- second-innings chases only
------------------------------------------------------------------------
Only innings == 2 (the chase) is modeled. A first-innings equivalent
("will this total be enough") is a different, harder problem -- there's
no fixed target to measure progress against -- and is intentionally left
out rather than bolted on with a weaker feature set. Ties and no-result
matches are excluded (no winner label). Matches decided under the D/L
method (`method == "D/L"`, 19 matches / ~1.6% of the dataset) are ALSO
excluded: this dataset has no revised-target/revised-overs data for
rain-affected matches, so treating a D/L match's target as the full
first-innings score over a full 20 overs would silently mislabel the
"runs needed" / "overs remaining" features for that match. Excluding
these 19 matches is a small, honest data-quality tradeoff -- see
docs/MODEL_CARD.md for the exact count.

------------------------------------------------------------------------
CRITICAL: MATCH-LEVEL, NOT BALL-LEVEL, TRAIN/TEST SPLITTING
------------------------------------------------------------------------
Every ball within a single match shares the same eventual winner and is
highly correlated with every other ball in that match. If train/test rows
were split at the ball level, the SAME match's balls would end up on both
sides of the split, letting the model implicitly "see" that match's
outcome during training and inflating test accuracy in a way that
wouldn't hold up on genuinely unseen matches. `build_live_state_dataset()`
returns a `match_id`-per-row `groups` array specifically so
`src/models/live_win_probability.py` can split and cross-validate at the
MATCH level (via `GroupKFold` / a match-level chronological split), never
the ball level. This is the single most important correctness property
of this module -- get it wrong and the reported accuracy is meaningless.

Usage:
    from src.features.live_state_engineering import build_live_state_dataset

    X, y, feature_names, encoders, groups = build_live_state_dataset()
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_all
from src.features.engineering import (
    CAREER_FORM_HALFLIFE,
    NEUTRAL_WIN_RATE,
    _build_team_match_log,
)
from src.utils.config import (
    LIVE_ENCODERS_PATH,
    LIVE_MODEL_FEATURES_PATH,
    ensure_directories_exist,
)
from src.utils.logger import get_logger

from sklearn.preprocessing import LabelEncoder

logger = get_logger(__name__)

# Standard T20 innings length. Verified against this dataset: every match
# in data/raw/matches.csv has `balls_per_over == 6` (checked 2026-07-11),
# so 20 overs x 6 balls = 120 legal balls is a safe constant rather than
# something that needs to be derived per-match.
BALLS_PER_OVER: int = 6
OVERS_PER_INNINGS: int = 20
TOTAL_LEGAL_BALLS: int = BALLS_PER_OVER * OVERS_PER_INNINGS

TARGET_COLUMN: str = "batting_team_wins"

FEATURE_COLUMNS: list[str] = [
    "batting_team_enc",
    "bowling_team_enc",
    "venue_enc",
    "target",
    "runs_scored",
    "runs_needed",
    "wickets_lost",
    "wickets_in_hand",
    "balls_bowled",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
    "run_rate_diff",
    "batting_team_career_win_pct",
    "bowling_team_career_win_pct",
]

CATEGORICAL_ENCODERS: list[str] = ["batting_team", "bowling_team", "venue"]


def _filter_eligible_chases(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict matches/deliveries to eligible second-innings chases: a
    decisive result, a known first-innings target, and NOT a D/L-affected
    match (see module docstring for why D/L matches are excluded).

    Args:
        matches_df: Cleaned matches DataFrame.
        deliveries_df: Cleaned deliveries DataFrame.

    Returns:
        Tuple of (eligible matches, deliveries restricted to innings==2
        of those eligible matches).
    """
    eligible = matches_df[
        (~matches_df["is_no_result"])
        & matches_df["first_innings_score"].notna()
        & (matches_df["method"] != "D/L")
    ].copy()

    n_excluded_dl = (matches_df["method"] == "D/L").sum()
    n_excluded_no_result = matches_df["is_no_result"].sum()
    logger.info(
        f"Live-state dataset: {len(eligible):,} eligible matches "
        f"(excluded {n_excluded_no_result} no-result/tied, "
        f"{n_excluded_dl} D/L-affected)."
    )

    eligible_ids = set(eligible["match_id"])
    chase_deliveries = deliveries_df[
        (deliveries_df["match_id"].isin(eligible_ids)) & (deliveries_df["innings"] == 2)
    ].copy()
    return eligible, chase_deliveries


def _build_ball_state(eligible_matches: pd.DataFrame, chase_deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct, for every ball of every eligible chase, the match state
    as it stood immediately BEFORE that ball was bowled (leakage-safe --
    the model must never see the outcome of the very ball it's meant to
    be predicting the winner "as of").

    Args:
        eligible_matches: Output of `_filter_eligible_chases()` (matches).
        chase_deliveries: Output of `_filter_eligible_chases()` (deliveries,
            innings==2 only).

    Returns:
        DataFrame with one row per ball: match_id, batting_team,
        bowling_team, venue, target, runs_scored, runs_needed,
        wickets_lost, wickets_in_hand, balls_bowled, balls_remaining,
        current_run_rate, required_run_rate, run_rate_diff, and the
        target label `batting_team_wins`.
    """
    df = chase_deliveries.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)

    is_legal_ball = ~df["extra_type"].isin(["wide", "no-ball"])
    df["_legal_ball"] = is_legal_ball.astype(int)

    grouped = df.groupby("match_id", group_keys=False)

    # Cumulative state BEFORE this ball = shift(1) of the running total
    # within each match (first ball of an innings has 0 runs, 0 wickets,
    # 0 balls bowled so far -- shift(1) on an empty prior window is NaN,
    # filled to 0 below).
    df["runs_scored"] = grouped["total_runs"].apply(lambda s: s.cumsum().shift(1)).reset_index(level=0, drop=True).fillna(0)
    df["wickets_lost"] = grouped["is_wicket"].apply(lambda s: s.astype(int).cumsum().shift(1)).reset_index(level=0, drop=True).fillna(0)
    df["balls_bowled"] = grouped["_legal_ball"].apply(lambda s: s.cumsum().shift(1)).reset_index(level=0, drop=True).fillna(0)

    target_map = eligible_matches.set_index("match_id")["first_innings_score"] + 1
    winner_map = eligible_matches.set_index("match_id")["winner"]
    venue_map = eligible_matches.set_index("match_id")["venue"]

    df["target"] = df["match_id"].map(target_map)
    df["venue"] = df["match_id"].map(venue_map)
    df["runs_needed"] = (df["target"] - df["runs_scored"]).clip(lower=0)
    df["wickets_in_hand"] = (10 - df["wickets_lost"]).clip(lower=0)
    df["balls_remaining"] = (TOTAL_LEGAL_BALLS - df["balls_bowled"]).clip(lower=0)

    overs_bowled = df["balls_bowled"] / BALLS_PER_OVER
    overs_remaining = df["balls_remaining"] / BALLS_PER_OVER
    df["current_run_rate"] = np.where(overs_bowled > 0, df["runs_scored"] / overs_bowled, 0.0)
    # required_run_rate is undefined at balls_remaining == 0 (last ball of
    # the innings) -- in that edge case there ARE no more overs left to
    # spread the requirement over, so fall back to a large sentinel value
    # scaled to how many runs are actually still needed, rather than
    # dividing by zero or silently using 0 (which would misleadingly imply
    # "no runs needed").
    df["required_run_rate"] = np.where(
        overs_remaining > 0, df["runs_needed"] / overs_remaining, df["runs_needed"] * BALLS_PER_OVER
    )
    df["run_rate_diff"] = df["current_run_rate"] - df["required_run_rate"]

    winner = df["match_id"].map(winner_map)
    df[TARGET_COLUMN] = (df["batting_team"] == winner).astype(int)

    keep_cols = [
        "match_id", "batting_team", "bowling_team", "venue",
        "target", "runs_scored", "runs_needed", "wickets_lost", "wickets_in_hand",
        "balls_bowled", "balls_remaining", "current_run_rate", "required_run_rate",
        "run_rate_diff", TARGET_COLUMN,
    ]
    return df[keep_cols]


def _add_team_strength_features(state_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend in each team's PRE-MATCH career win pct (as of just before this
    match, leakage-safe, same recency-weighted computation as the
    pre-match model -- reuses `src.features.engineering._build_team_match_log`
    directly rather than reimplementing it) as two extra features:
    `batting_team_career_win_pct` and `bowling_team_career_win_pct`.

    Rationale: in-match state (score, wickets, overs) dominates the
    prediction, but team strength still carries a little residual signal
    even mid-chase (e.g. a stronger bowling attack defending a total makes
    the same "10 runs off 6 balls" situation slightly less favorable for
    the batting side than it would be against a weaker attack). This is a
    genuinely small effect expected to matter far less than the live
    match-state features -- included for completeness/consistency with
    the pre-match model, not because it's expected to be a major driver
    here.

    Args:
        state_df: Output of `_build_ball_state()`.
        matches_df: Cleaned matches DataFrame (used to build the
            chronological team-strength log).

    Returns:
        `state_df` with two additional columns.
    """
    log = _build_team_match_log(matches_df)
    # For every match, the team's career_win_pct "as of just before this
    # match" is exactly the (leakage-safe) career_win_pct already computed
    # by _add_rolling_team_stats in engineering.py -- but that function
    # also computes venue/recent-form stats we don't need here, so we
    # recompute only the piece we need directly for clarity and to avoid
    # an unnecessary dependency on venue-shrinkage internals.
    log = log.sort_values(["date", "match_id"]).reset_index(drop=True)
    log["career_win_pct"] = (
        log.groupby("team", group_keys=False)["is_winner"]
        .apply(lambda s: s.shift(1).ewm(halflife=CAREER_FORM_HALFLIFE, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(NEUTRAL_WIN_RATE)
    )
    strength_lookup = log[["match_id", "team", "career_win_pct"]]

    state_df = state_df.copy()
    state_df = state_df.merge(
        strength_lookup.rename(columns={"team": "batting_team", "career_win_pct": "batting_team_career_win_pct"}),
        on=["match_id", "batting_team"], how="left",
    )
    state_df = state_df.merge(
        strength_lookup.rename(columns={"team": "bowling_team", "career_win_pct": "bowling_team_career_win_pct"}),
        on=["match_id", "bowling_team"], how="left",
    )
    state_df["batting_team_career_win_pct"] = state_df["batting_team_career_win_pct"].fillna(NEUTRAL_WIN_RATE)
    state_df["bowling_team_career_win_pct"] = state_df["bowling_team_career_win_pct"].fillna(NEUTRAL_WIN_RATE)
    return state_df


def _encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode `batting_team`, `bowling_team`, `venue` into `_enc`
    integer columns, mirroring `src.features.engineering._encode_categoricals`
    (kept as a separate copy rather than imported, since this module's
    CATEGORICAL_ENCODERS list differs from the pre-match model's).

    Args:
        df: DataFrame containing the raw categorical columns.
        fit: If True, fit new encoders. If False, reuse `encoders`,
            mapping unseen categories to a trailing "unknown" bucket.
        encoders: Required when `fit=False`.

    Returns:
        Tuple of (DataFrame with `_enc` columns added, encoders dict).
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
            known = set(le.classes_)
            unknown_label = len(le.classes_)
            df[f"{col}_enc"] = df[col].astype(str).apply(
                lambda v: int(le.transform([v])[0]) if v in known else unknown_label
            )

    return df, encoders


def build_live_state_dataset(save: bool = True) -> tuple[pd.DataFrame, pd.Series, list[str], dict, pd.Series]:
    """
    Run the full live-state feature engineering pipeline end-to-end:
    load + clean raw data, restrict to eligible second-innings chases,
    reconstruct ball-by-ball match state, blend in team-strength features,
    and encode categoricals.

    Args:
        save: If True (default), persist the feature matrix to
            `LIVE_MODEL_FEATURES_PATH` and the fitted encoders to
            `LIVE_ENCODERS_PATH`.

    Returns:
        Tuple of:
            X: Feature matrix (DataFrame), columns = FEATURE_COLUMNS.
            y: Target Series (1 if the batting/chasing team won).
            feature_names: FEATURE_COLUMNS (convenience passthrough).
            encoders: Dict of fitted LabelEncoders.
            groups: Series of `match_id`, ALIGNED with X/y -- REQUIRED for
                match-level (not ball-level) cross-validation and
                train/test splitting. See module docstring.

    Example:
        >>> X, y, feature_names, encoders, groups = build_live_state_dataset()
        >>> X.shape[0] == y.shape[0] == groups.shape[0]
        True
    """
    logger.info("Building in-match (live) win-probability feature dataset...")
    ensure_directories_exist()

    cleaned = preprocess_all(save=False)
    matches_df, deliveries_df = cleaned["matches"], cleaned["deliveries"]

    eligible_matches, chase_deliveries = _filter_eligible_chases(matches_df, deliveries_df)
    state_df = _build_ball_state(eligible_matches, chase_deliveries)
    state_df = _add_team_strength_features(state_df, matches_df)
    state_df, encoders = _encode_categoricals(state_df, fit=True)

    X = state_df[FEATURE_COLUMNS].copy()
    y = state_df[TARGET_COLUMN].copy()
    groups = state_df["match_id"].copy()

    logger.info(
        f"Live-state dataset built: {X.shape[0]:,} balls across "
        f"{groups.nunique():,} matches x {X.shape[1]} features "
        f"(target balance: {y.mean():.1%} batting-team wins)."
    )

    if save:
        out = X.copy()
        out[TARGET_COLUMN] = y
        out["match_id"] = groups
        out.to_csv(LIVE_MODEL_FEATURES_PATH, index=False)
        joblib.dump(encoders, LIVE_ENCODERS_PATH)
        logger.info(f"Saved live-state feature matrix to {LIVE_MODEL_FEATURES_PATH}")
        logger.info(f"Saved fitted live-state encoders to {LIVE_ENCODERS_PATH}")

    return X, y, FEATURE_COLUMNS, encoders, groups


def get_match_state_timeline(match_id: str, matches_df: pd.DataFrame | None = None, deliveries_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """
    Reconstruct the full ball-by-ball second-innings state timeline for a
    SINGLE historical match, including wicket-fall flags and over/ball
    numbers -- used by the Streamlit "Live Win Probability" page to replay
    a real chase and by `src.models.live_win_probability.
    compute_match_win_probability_timeline()` to attach model predictions
    to it for the win-probability timeline chart
    (`src.visualization.plots.plot_win_probability_timeline`).

    This reuses the same `_build_ball_state()` logic as
    `build_live_state_dataset()` (so a replayed match's state is computed
    identically to how it was computed during training), but returns the
    UNENCODED, per-ball DataFrame for a single match rather than the full
    encoded training matrix -- the caller is responsible for encoding
    categoricals if feeding rows into a trained model.

    Args:
        match_id: The match to reconstruct.
        matches_df: Optional pre-loaded cleaned matches DataFrame. If
            None, recomputed via `preprocess_all()`.
        deliveries_df: Optional pre-loaded cleaned deliveries DataFrame.
            If None, recomputed via `preprocess_all()`.

    Returns:
        DataFrame with one row per ball (over, ball, batting_team,
        bowling_team, venue, target, runs_scored, runs_needed,
        wickets_lost, wickets_in_hand, balls_bowled, balls_remaining,
        current_run_rate, required_run_rate, run_rate_diff, is_wicket,
        batting_team_wins), or None if `match_id` is not an eligible
        second-innings chase (no result, no first-innings score on
        record, or D/L-affected -- see `_filter_eligible_chases()`).
    """
    if matches_df is None or deliveries_df is None:
        cleaned = preprocess_all(save=False)
        matches_df = matches_df if matches_df is not None else cleaned["matches"]
        deliveries_df = deliveries_df if deliveries_df is not None else cleaned["deliveries"]

    eligible_matches, chase_deliveries = _filter_eligible_chases(matches_df, deliveries_df)
    if match_id not in set(eligible_matches["match_id"]):
        return None

    single_match = eligible_matches[eligible_matches["match_id"] == match_id]
    single_deliveries = chase_deliveries[chase_deliveries["match_id"] == match_id]

    state_df = _build_ball_state(single_match, single_deliveries)
    state_df = _add_team_strength_features(state_df, matches_df)

    # Re-attach over/ball and is_wicket, which _build_ball_state() drops
    # from its returned `keep_cols` (they're not model features, but are
    # needed for the replay chart's x-axis and wicket markers).
    raw = single_deliveries.sort_values(["over", "ball"]).reset_index(drop=True)
    state_df = state_df.reset_index(drop=True)
    state_df["over"] = raw["over"]
    state_df["ball"] = raw["ball"]
    state_df["is_wicket"] = raw["is_wicket"].astype(bool)

    return state_df


if __name__ == "__main__":
    # Allows running `python -m src.features.live_state_engineering`
    # directly as a quick sanity check that the pipeline runs end-to-end.
    X, y, feature_names, encoders, groups = build_live_state_dataset()
    print(f"X shape: {X.shape}")
    print(f"y balance: {y.mean():.3f}")
    print(f"unique matches: {groups.nunique()}")
    print(X.head())
