"""
team_analytics.py
==================
Team-level analytics for the Cricket Analytics & Match Prediction Platform.

This module computes descriptive (NOT predictive) statistics about teams
from the cleaned matches DataFrame -- win/loss records, head-to-head
records, toss impact, venue performance, and home/away-style splits.
It deliberately has nothing to do with the ML pipeline in `src/models/` or
`src/features/`: this is pure analytics for the "Team Analytics" dashboard
page, computed over the ENTIRE historical record (no leakage concerns,
since nothing here is used as a model feature).

Usage:
    from src.data.preprocess import preprocess_all
    from src.analytics.team_analytics import compute_team_summary

    matches_df = preprocess_all(save=False)["matches"]
    summary = compute_team_summary(matches_df)
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from src.data.loader import DataValidationError
from src.utils.config import (
    CITY_COLUMN,
    SEASON_COLUMN,
    TARGET_COLUMN as WINNER_COLUMN,
    TEAM1_COLUMN,
    TEAM2_COLUMN,
    TOSS_DECISION_COLUMN,
    TOSS_WINNER_COLUMN,
    VENUE_COLUMN,
    WIN_BY_COLUMN,
    WIN_MARGIN_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "compute_team_summary",
    "compute_head_to_head",
    "compute_toss_impact",
    "compute_team_venue_performance",
    "compute_season_wise_performance",
    "compute_home_away_style_split",
    "compute_recent_form",
    "compute_team_profile",
]

# `is_no_result` is derived by `src/data/preprocess.py` (it does not exist
# in the raw Kaggle schema, so it has no constant in `src/utils/config.py`).
# Named here once so the rest of this module never repeats the literal.
IS_NO_RESULT_COLUMN: str = "is_no_result"
DATE_COLUMN: str = "date"

_REQUIRED_MATCH_COLUMNS: frozenset[str] = frozenset({
    TEAM1_COLUMN, TEAM2_COLUMN, WINNER_COLUMN, IS_NO_RESULT_COLUMN,
})


class RecordSplit(TypedDict):
    """A single matches_played/wins/win_pct bucket, shared by several functions."""

    matches_played: int
    wins: int
    win_pct: float


class HeadToHeadRecord(TypedDict):
    """Return shape of `compute_head_to_head()`."""

    total_matches: int
    team_a_wins: int
    team_b_wins: int
    no_results: int
    matches: pd.DataFrame


class TossImpact(TypedDict):
    """Return shape of `compute_toss_impact()`."""

    overall_toss_win_match_win_pct: float
    by_decision: pd.DataFrame


class HomeAwaySplit(TypedDict):
    """Return shape of `compute_home_away_style_split()`."""

    home: RecordSplit
    away: RecordSplit


class TeamProfile(TypedDict):
    """Return shape of `compute_team_profile()`."""

    summary: dict[str, Any] | None
    venue_performance: pd.DataFrame
    season_wise_performance: pd.DataFrame
    home_away_split: HomeAwaySplit | None
    recent_form: RecordSplit


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
            f"output of `preprocess_all()`? (`is_no_result` in particular is "
            f"only added during preprocessing.)"
        )


def _safe_win_pct(wins: pd.Series, decided: pd.Series) -> pd.Series:
    """
    Vectorized win-percentage calculation that returns 0.0 (not NaN/inf)
    wherever a team/venue/season has zero decided matches.

    Args:
        wins: Series of win counts.
        decided: Series of decided-match counts (matches_played - no_results).

    Returns:
        Series of win percentages, 0-100, rounded to 1 decimal place.
    """
    decided_safe = decided.replace(0, pd.NA)
    return (wins / decided_safe * 100).fillna(0).round(1)


def _team_matches_with_win_flag(matches_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Filter to every match a single team played in, with a boolean `is_win`
    column added.

    Centralizes a filter+flag pattern that was previously duplicated
    (with identical logic, copy-pasted) across `compute_team_venue_performance`,
    `compute_season_wise_performance`, and `compute_home_away_style_split`.

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name to filter to.

    Returns:
        Copy of the matches this team played in, with an added `is_win`
        boolean column (True only for decided matches this team won).
    """
    team_matches = matches_df[
        (matches_df[TEAM1_COLUMN] == team) | (matches_df[TEAM2_COLUMN] == team)
    ].copy()
    team_matches["is_win"] = (
        (~team_matches[IS_NO_RESULT_COLUMN]) & (team_matches[WINNER_COLUMN] == team)
    )
    if team_matches.empty:
        logger.warning(f"No matches found for team '{team}'.")
    return team_matches


def _summarize_record(subset: pd.DataFrame) -> RecordSplit:
    """
    Reduce a (possibly already win-flagged) subset of matches to a single
    matches_played/wins/win_pct record.

    Args:
        subset: A matches DataFrame slice that already has an `is_win`
            boolean column (e.g. produced by `_team_matches_with_win_flag`).

    Returns:
        Dict with "matches_played", "wins", "win_pct" (0.0 if there are
        no decided matches in the subset).
    """
    decided = len(subset) - int(subset[IS_NO_RESULT_COLUMN].sum())
    wins = int(subset["is_win"].sum())
    win_pct = round(wins / decided * 100, 1) if decided > 0 else 0.0
    return {"matches_played": int(len(subset)), "wins": wins, "win_pct": win_pct}


def compute_team_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one row per team with career win/loss totals and win percentage.

    Args:
        matches_df: Cleaned matches DataFrame (must include `is_no_result`).

    Returns:
        DataFrame with columns: team, matches_played, wins, losses,
        no_results, win_pct (0-100), sorted by win_pct descending.

    Raises:
        DataValidationError: If `matches_df` is missing required columns.
    """
    _validate_match_columns(matches_df)
    columns = ["team", "matches_played", "wins", "losses", "no_results", "win_pct"]
    if matches_df.empty:
        logger.warning("compute_team_summary() called with an empty matches DataFrame.")
        return pd.DataFrame(columns=columns)

    # Vectorized "long" reshape: each match contributes one row per side
    # (team1's perspective, team2's perspective) instead of iterating
    # `matches_df.iterrows()` in Python, which previously rebuilt a
    # 2 x len(matches_df) list via a per-row loop -- correct, but the
    # slowest possible way to do this in pandas.
    sides = []
    for team_col in (TEAM1_COLUMN, TEAM2_COLUMN):
        side = pd.DataFrame({
            "team": matches_df[team_col],
            IS_NO_RESULT_COLUMN: matches_df[IS_NO_RESULT_COLUMN],
            "is_win": (~matches_df[IS_NO_RESULT_COLUMN]) & (matches_df[WINNER_COLUMN] == matches_df[team_col]),
        })
        sides.append(side)
    played = pd.concat(sides, ignore_index=True)

    summary = played.groupby("team").agg(
        matches_played=("team", "count"),
        no_results=(IS_NO_RESULT_COLUMN, "sum"),
        wins=("is_win", "sum"),
    ).reset_index()
    summary["losses"] = summary["matches_played"] - summary["wins"] - summary["no_results"]
    decided = summary["matches_played"] - summary["no_results"]
    summary["win_pct"] = _safe_win_pct(summary["wins"], decided)
    summary = summary.sort_values("win_pct", ascending=False).reset_index(drop=True)
    logger.info(f"Computed team summary for {len(summary)} teams.")
    return summary[columns]


def compute_head_to_head(matches_df: pd.DataFrame, team_a: str, team_b: str) -> HeadToHeadRecord:
    """
    Compute the complete head-to-head record between two specific teams.

    Args:
        matches_df: Cleaned matches DataFrame.
        team_a: First team's name.
        team_b: Second team's name.

    Returns:
        Dict with keys: "total_matches", "team_a_wins", "team_b_wins",
        "no_results", and "matches" (DataFrame of the individual matches
        between these two teams, most recent first).

    Raises:
        DataValidationError: If `matches_df` is missing required columns.
    """
    _validate_match_columns(matches_df)
    mask = (
        ((matches_df[TEAM1_COLUMN] == team_a) & (matches_df[TEAM2_COLUMN] == team_b))
        | ((matches_df[TEAM1_COLUMN] == team_b) & (matches_df[TEAM2_COLUMN] == team_a))
    )
    h2h_matches = matches_df[mask].copy()
    if h2h_matches.empty:
        logger.warning(f"No head-to-head matches found between '{team_a}' and '{team_b}'.")

    h2h_matches[DATE_COLUMN] = pd.to_datetime(h2h_matches[DATE_COLUMN])
    h2h_matches = h2h_matches.sort_values(DATE_COLUMN, ascending=False)

    team_a_wins = int((h2h_matches[WINNER_COLUMN] == team_a).sum())
    team_b_wins = int((h2h_matches[WINNER_COLUMN] == team_b).sum())
    no_results = int(h2h_matches[IS_NO_RESULT_COLUMN].sum())

    return {
        "total_matches": int(len(h2h_matches)),
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "no_results": no_results,
        "matches": h2h_matches[
            [DATE_COLUMN, SEASON_COLUMN, VENUE_COLUMN, TOSS_WINNER_COLUMN,
             TOSS_DECISION_COLUMN, WINNER_COLUMN, WIN_BY_COLUMN, WIN_MARGIN_COLUMN]
        ],
    }


def compute_toss_impact(matches_df: pd.DataFrame) -> TossImpact:
    """
    Quantify whether winning the toss correlates with winning the match,
    overall and broken down by the toss decision (bat vs field first).

    Args:
        matches_df: Cleaned matches DataFrame.

    Returns:
        Dict with keys:
            "overall_toss_win_match_win_pct": % of decided matches where the
                toss winner also won the match.
            "by_decision": DataFrame with columns toss_decision,
                matches, toss_winner_also_match_winner_pct.

    Raises:
        DataValidationError: If `matches_df` is missing required columns.
    """
    _validate_match_columns(matches_df)
    decided = matches_df[~matches_df[IS_NO_RESULT_COLUMN]].copy()
    decided["toss_winner_won_match"] = decided[TOSS_WINNER_COLUMN] == decided[WINNER_COLUMN]

    overall_pct = round(decided["toss_winner_won_match"].mean() * 100, 1) if len(decided) else 0.0

    by_decision = (
        decided.groupby(TOSS_DECISION_COLUMN)
        .agg(
            matches=("toss_winner_won_match", "count"),
            toss_winner_also_match_winner_pct=("toss_winner_won_match", lambda s: round(s.mean() * 100, 1)),
        )
        .reset_index()
    )
    logger.info(f"Toss impact: toss winner also won {overall_pct}% of {len(decided)} decided matches.")

    return {
        "overall_toss_win_match_win_pct": overall_pct,
        "by_decision": by_decision,
    }


def compute_team_venue_performance(matches_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Compute a single team's win percentage at every venue it has played at.

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name to compute venue performance for.

    Returns:
        DataFrame with columns: venue, matches_played, wins, win_pct,
        sorted by matches_played descending. Venues with zero decided
        matches (all no-results) show win_pct as 0. Empty (but
        correctly-columned) if the team has no matches at all.
    """
    _validate_match_columns(matches_df)
    columns = ["venue", "matches_played", "wins", "win_pct"]
    team_matches = _team_matches_with_win_flag(matches_df, team)
    if team_matches.empty:
        return pd.DataFrame(columns=columns)

    summary = team_matches.groupby(VENUE_COLUMN).agg(
        matches_played=(VENUE_COLUMN, "count"),
        no_results=(IS_NO_RESULT_COLUMN, "sum"),
        wins=("is_win", "sum"),
    ).reset_index().rename(columns={VENUE_COLUMN: "venue"})
    decided = summary["matches_played"] - summary["no_results"]
    summary["win_pct"] = _safe_win_pct(summary["wins"], decided)
    summary = summary.sort_values("matches_played", ascending=False).reset_index(drop=True)
    return summary[columns]


def compute_season_wise_performance(matches_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Compute a single team's win percentage in each IPL season it played in.

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name to compute season-wise performance for.

    Returns:
        DataFrame with columns: season, matches_played, wins, win_pct,
        sorted by season ascending (for plotting a trend line over time).
        Empty (but correctly-columned) if the team has no matches at all.
    """
    _validate_match_columns(matches_df)
    columns = ["season", "matches_played", "wins", "win_pct"]
    team_matches = _team_matches_with_win_flag(matches_df, team)
    if team_matches.empty:
        return pd.DataFrame(columns=columns)

    summary = team_matches.groupby(SEASON_COLUMN).agg(
        matches_played=(SEASON_COLUMN, "count"),
        no_results=(IS_NO_RESULT_COLUMN, "sum"),
        wins=("is_win", "sum"),
    ).reset_index().rename(columns={SEASON_COLUMN: "season"})
    decided = summary["matches_played"] - summary["no_results"]
    summary["win_pct"] = _safe_win_pct(summary["wins"], decided)
    summary = summary.sort_values("season").reset_index(drop=True)
    return summary[columns]


def compute_home_away_style_split(matches_df: pd.DataFrame, team: str, team_home_city: str) -> HomeAwaySplit:
    """
    Split a team's results into matches played in its "home" city vs all
    other ("away") cities, as a simple proxy for home-ground advantage
    (the dataset has no explicit home/away flag, so this is approximated
    using the city of the venue).

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name.
        team_home_city: City name considered "home" for this team (e.g.
            "Mumbai" for Mumbai Indians). The Streamlit page supplies this
            from a small team-to-city lookup.

    Returns:
        Dict with keys "home" and "away", each mapping to a dict with
        "matches_played", "wins", "win_pct".
    """
    _validate_match_columns(matches_df)
    team_matches = _team_matches_with_win_flag(matches_df, team)

    home_subset = team_matches[team_matches[CITY_COLUMN] == team_home_city]
    away_subset = team_matches[team_matches[CITY_COLUMN] != team_home_city]

    if home_subset.empty:
        logger.warning(f"'{team}' has no recorded matches in its home city '{team_home_city}'.")

    return {"home": _summarize_record(home_subset), "away": _summarize_record(away_subset)}


def compute_recent_form(matches_df: pd.DataFrame, team: str, last_n: int = 5) -> RecordSplit:
    """
    Compute a team's win percentage over its most recent `last_n` matches
    -- a simple "current form" indicator for the dashboard, distinct from
    the leakage-safe rolling-form FEATURE built in
    `src/features/engineering.py` (that one is computed per-match, as of
    each match's date, for model training; this one is a single snapshot
    as of "today", for display purposes only).

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name.
        last_n: Number of most recent matches to consider. Defaults to 5.

    Returns:
        Dict with "matches_played" (min(last_n, total matches played)),
        "wins", "win_pct".
    """
    _validate_match_columns(matches_df)
    team_matches = _team_matches_with_win_flag(matches_df, team)
    if team_matches.empty:
        return {"matches_played": 0, "wins": 0, "win_pct": 0.0}

    team_matches[DATE_COLUMN] = pd.to_datetime(team_matches[DATE_COLUMN])
    recent = team_matches.sort_values(DATE_COLUMN, ascending=False).head(last_n)
    return _summarize_record(recent)


def compute_team_profile(
    matches_df: pd.DataFrame,
    team: str,
    team_home_city: str | None = None,
    recent_form_n: int = 5,
) -> TeamProfile:
    """
    Build a single team's full profile in one call: career summary, venue
    performance, season-wise trend, optional home/away split, and recent
    form -- the team-level analogue of
    `venue_analytics.compute_venue_profile()` and
    `player_analytics.compute_player_profile()`, for the Team Analytics
    dashboard page to fetch everything about one team in a single call
    instead of separately invoking five functions.

    Args:
        matches_df: Cleaned matches DataFrame.
        team: Team name to profile.
        team_home_city: Optional "home" city for the home/away split (see
            `compute_home_away_style_split()`). If omitted, "home_away_split"
            is None and not computed.
        recent_form_n: Number of most recent matches for the recent-form
            snapshot. Defaults to 5.

    Returns:
        Dict with keys "summary" (single-row dict from
        `compute_team_summary()`, or None if the team has no matches),
        "venue_performance", "season_wise_performance" (DataFrames),
        "home_away_split" (dict or None), and "recent_form" (dict).
    """
    summary_df = compute_team_summary(matches_df)
    summary_row = summary_df[summary_df["team"] == team]
    summary = summary_row.iloc[0].to_dict() if not summary_row.empty else None
    if summary is None:
        logger.warning(f"compute_team_profile(): '{team}' has no record in the team summary.")

    home_away_split = (
        compute_home_away_style_split(matches_df, team, team_home_city)
        if team_home_city is not None
        else None
    )

    return {
        "summary": summary,
        "venue_performance": compute_team_venue_performance(matches_df, team),
        "season_wise_performance": compute_season_wise_performance(matches_df, team),
        "home_away_split": home_away_split,
        "recent_form": compute_recent_form(matches_df, team, last_n=recent_form_n),
    }
