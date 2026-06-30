"""
venue_analytics.py
===================
Venue-level analytics for the Cricket Analytics & Match Prediction
Platform.

Computes descriptive statistics about IPL venues -- average first/second
innings scores, how often the team batting first wins (a useful proxy for
"is this a chasing ground or a defending ground"), highest/lowest totals,
and which teams have played there most.

Usage:
    from src.data.preprocess import preprocess_all
    from src.analytics.venue_analytics import compute_venue_summary

    matches_df = preprocess_all(save=False)["matches"]
    summary = compute_venue_summary(matches_df)
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from src.data.loader import DataValidationError
from src.utils.config import (
    CITY_COLUMN,
    DATE_COLUMN,
    FIRST_INNINGS_SCORE_COLUMN,
    SEASON_COLUMN,
    SECOND_INNINGS_SCORE_COLUMN,
    TARGET_COLUMN as WINNER_COLUMN,
    TEAM1_COLUMN,
    TEAM2_COLUMN,
    TOSS_DECISION_COLUMN,
    TOSS_WINNER_COLUMN,
    VENUE_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "derive_bat_first_team",
    "compute_venue_summary",
    "compute_venue_profile",
    "compute_city_summary",
]

# `is_no_result` is derived by `src/data/preprocess.py` (it does not exist
# in the raw Kaggle schema, so it has no constant in `src/utils/config.py`).
IS_NO_RESULT_COLUMN: str = "is_no_result"

_REQUIRED_MATCH_COLUMNS: frozenset[str] = frozenset({
    VENUE_COLUMN, CITY_COLUMN, TEAM1_COLUMN, TEAM2_COLUMN, WINNER_COLUMN,
    IS_NO_RESULT_COLUMN, TOSS_WINNER_COLUMN, TOSS_DECISION_COLUMN,
    FIRST_INNINGS_SCORE_COLUMN, SECOND_INNINGS_SCORE_COLUMN,
})


class VenueProfile(TypedDict):
    """Return shape of `compute_venue_profile()`."""

    summary: dict[str, Any] | None
    most_frequent_teams: pd.DataFrame
    highest_scoring_matches: pd.DataFrame
    lowest_scoring_matches: pd.DataFrame


def _validate_match_columns(matches_df: pd.DataFrame) -> None:
    """
    Verify that `matches_df` has the columns this module depends on.

    Args:
        matches_df: Matches DataFrame to validate.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = _REQUIRED_MATCH_COLUMNS - set(matches_df.columns)
    if missing:
        raise DataValidationError(
            f"matches DataFrame is missing required columns: {sorted(missing)}. "
            f"Did you pass the raw (uncleaned) matches.csv instead of the "
            f"output of `preprocess_all()`?"
        )


def derive_bat_first_team(matches_df: pd.DataFrame) -> pd.Series:
    """
    Derive which team actually batted first in each match, from the toss
    result -- NOT from `team1`/`team2` column order.

    This exists to fix a real correctness bug: an earlier version of this
    module assumed `team1` was always the side that batted first. That
    assumption is FALSE for this dataset -- cross-checking against
    `matches_clean.csv` shows `team1` only batted first in roughly half
    of all matches (the other half, `team1` won the toss and chose to
    field, or `team2` won the toss and chose to bat). Using `team1_won`
    as a stand-in for "the side batting first won" silently inverted the
    correct answer for about half the rows in every venue's stats, with
    individual venues affected by as much as 20+ percentage points even
    though the league-wide average happened to look only mildly off
    (errors partly canceled out in aggregate, which is what made the bug
    easy to miss).

    The correct rule, from the toss result:
        - If `team1` won the toss and chose to bat -> `team1` bats first.
        - If `team2` won the toss and chose to field -> `team1` bats first.
        - Otherwise -> `team2` bats first.

    Args:
        matches_df: Cleaned matches DataFrame (must include `toss_winner`,
            `toss_decision`, `team1`, `team2`).

    Returns:
        Series (same index as `matches_df`) of team names, the side that
        batted first in each match. Rows with missing/unrecognized toss
        data return `pd.NA` rather than a guessed value.
    """
    team1_won_toss = matches_df[TOSS_WINNER_COLUMN] == matches_df[TEAM1_COLUMN]
    team1_bats_first = (
        (team1_won_toss & (matches_df[TOSS_DECISION_COLUMN] == "bat"))
        | (~team1_won_toss & (matches_df[TOSS_DECISION_COLUMN] == "field"))
    )

    unrecognized = matches_df[TOSS_DECISION_COLUMN].isna() | matches_df[TOSS_WINNER_COLUMN].isna()
    if unrecognized.any():
        logger.warning(
            f"{int(unrecognized.sum())} matches have missing toss_winner/toss_decision -- "
            f"their bat-first team cannot be determined and will be left as NA."
        )

    bat_first_team = matches_df[TEAM1_COLUMN].where(team1_bats_first, matches_df[TEAM2_COLUMN])
    return bat_first_team.mask(unrecognized, pd.NA)


def compute_venue_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one row per venue with match counts, average first/second
    innings scores, and how often the side batting first goes on to win.

    Args:
        matches_df: Cleaned matches DataFrame.

    Returns:
        DataFrame with columns: venue, city, matches_played,
        avg_first_innings_score, avg_second_innings_score,
        bat_first_win_pct, highest_total, lowest_total -- sorted by
        matches_played descending.

    Raises:
        DataValidationError: If `matches_df` is missing required columns.
    """
    _validate_match_columns(matches_df)
    columns = [
        "venue", "city", "matches_played", "avg_first_innings_score",
        "avg_second_innings_score", "bat_first_win_pct", "highest_total", "lowest_total",
    ]
    if matches_df.empty:
        logger.warning("compute_venue_summary() called with an empty matches DataFrame.")
        return pd.DataFrame(columns=columns)

    df = matches_df.copy()
    bat_first_team = derive_bat_first_team(df)
    df["bat_first_won"] = (~df[IS_NO_RESULT_COLUMN]) & (df[WINNER_COLUMN] == bat_first_team)

    grouped = df.groupby(VENUE_COLUMN)
    summary = grouped.agg(
        city=(CITY_COLUMN, "first"),
        matches_played=(VENUE_COLUMN, "count"),
        avg_first_innings_score=(FIRST_INNINGS_SCORE_COLUMN, "mean"),
        avg_second_innings_score=(SECOND_INNINGS_SCORE_COLUMN, "mean"),
        highest_total=(FIRST_INNINGS_SCORE_COLUMN, "max"),
        lowest_total=(FIRST_INNINGS_SCORE_COLUMN, "min"),
        no_results=(IS_NO_RESULT_COLUMN, "sum"),
        bat_first_wins=("bat_first_won", "sum"),
        distinct_cities=(CITY_COLUMN, "nunique"),
    ).reset_index().rename(columns={VENUE_COLUMN: "venue"})

    inconsistent_city = summary[summary["distinct_cities"] > 1]
    if not inconsistent_city.empty:
        logger.warning(
            f"{len(inconsistent_city)} venue(s) are associated with more than one "
            f"city name (possible inconsistent naming in the data): "
            f"{inconsistent_city['venue'].tolist()}"
        )

    decided = summary["matches_played"] - summary["no_results"]
    summary["bat_first_win_pct"] = (
        summary["bat_first_wins"] / decided.replace(0, pd.NA) * 100
    ).fillna(0).round(1)
    summary["avg_first_innings_score"] = summary["avg_first_innings_score"].round(1)
    summary["avg_second_innings_score"] = summary["avg_second_innings_score"].round(1)

    summary = summary.sort_values("matches_played", ascending=False).reset_index(drop=True)
    logger.info(f"Computed venue summary for {len(summary)} venues.")
    return summary[columns]


def compute_venue_profile(
    matches_df: pd.DataFrame,
    venue: str,
    venue_summary: pd.DataFrame | None = None,
) -> VenueProfile:
    """
    Build a detailed profile for a single venue: summary stats, the teams
    that have played there most often, and the highest/lowest-scoring
    matches on record there.

    Args:
        matches_df: Cleaned matches DataFrame.
        venue: Venue name to profile.
        venue_summary: Optionally, a summary already computed via
            `compute_venue_summary(matches_df)`. Pass this in when
            profiling many venues in a loop to avoid recomputing the
            league-wide summary -- an O(n) groupby -- on every call.

    Returns:
        Dict with keys:
            "summary": single-row dict from `compute_venue_summary()`, or
                None if the venue has no matches.
            "most_frequent_teams": DataFrame of team -> appearances at
                this venue, sorted descending.
            "highest_scoring_matches": top-5 matches by first_innings_score
                at this venue.
            "lowest_scoring_matches": bottom-5 matches by first_innings_score
                at this venue.
    """
    _validate_match_columns(matches_df)
    empty_profile: VenueProfile = {
        "summary": None,
        "most_frequent_teams": pd.DataFrame(columns=["team", "appearances"]),
        "highest_scoring_matches": pd.DataFrame(),
        "lowest_scoring_matches": pd.DataFrame(),
    }

    venue_matches = matches_df[matches_df[VENUE_COLUMN] == venue].copy()
    if venue_matches.empty:
        logger.warning(f"No matches found for venue '{venue}'.")
        return empty_profile

    summary_df = venue_summary if venue_summary is not None else compute_venue_summary(matches_df)
    summary_row_df = summary_df[summary_df["venue"] == venue]
    if summary_row_df.empty:
        logger.warning(f"Venue '{venue}' found in matches_df but missing from the venue summary.")
        return empty_profile
    summary_row = summary_row_df.iloc[0].to_dict()

    team_counts = (
        pd.concat([venue_matches[TEAM1_COLUMN], venue_matches[TEAM2_COLUMN]])
        .value_counts()
        .rename_axis("team")
        .reset_index(name="appearances")
    )

    display_cols = [
        DATE_COLUMN, SEASON_COLUMN, TEAM1_COLUMN, TEAM2_COLUMN,
        FIRST_INNINGS_SCORE_COLUMN, SECOND_INNINGS_SCORE_COLUMN, WINNER_COLUMN,
    ]
    highest = venue_matches.sort_values(FIRST_INNINGS_SCORE_COLUMN, ascending=False).head(5)[display_cols]
    lowest = venue_matches.sort_values(FIRST_INNINGS_SCORE_COLUMN, ascending=True).head(5)[display_cols]

    return {
        "summary": summary_row,
        "most_frequent_teams": team_counts,
        "highest_scoring_matches": highest,
        "lowest_scoring_matches": lowest,
    }


def compute_city_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll venue-level stats up to the city level (some cities host multiple
    venues across IPL history, e.g. relocated stadiums).

    Args:
        matches_df: Cleaned matches DataFrame.

    Returns:
        DataFrame with columns: city, matches_played, venues_count,
        avg_first_innings_score, sorted by matches_played descending.

    Raises:
        DataValidationError: If `matches_df` is missing required columns.
    """
    _validate_match_columns(matches_df)
    columns = ["city", "matches_played", "venues_count", "avg_first_innings_score"]
    if matches_df.empty:
        logger.warning("compute_city_summary() called with an empty matches DataFrame.")
        return pd.DataFrame(columns=columns)

    summary = matches_df.groupby(CITY_COLUMN).agg(
        matches_played=(CITY_COLUMN, "count"),
        venues_count=(VENUE_COLUMN, "nunique"),
        avg_first_innings_score=(FIRST_INNINGS_SCORE_COLUMN, "mean"),
    ).reset_index().rename(columns={CITY_COLUMN: "city"})
    summary["avg_first_innings_score"] = summary["avg_first_innings_score"].round(1)
    summary = summary.sort_values("matches_played", ascending=False).reset_index(drop=True)
    logger.info(f"Computed city summary for {len(summary)} cities.")
    return summary[columns]
