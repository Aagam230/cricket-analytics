"""
player_analytics.py
====================
Player-level analytics for the Cricket Analytics & Match Prediction
Platform.

Computes batting and bowling leaderboards and per-player career summaries
from the ball-by-ball `deliveries.csv` data, joined against `players.csv`
for metadata (nationality, role) and `matches.csv` for season context.

Like `team_analytics.py` and `venue_analytics.py`, this module is pure,
descriptive analytics for the Streamlit "Player Analytics" dashboard page.
It has nothing to do with the ML pipeline in `src/models/` or
`src/features/`: nothing computed here is used as a model feature, so
there are no temporal-leakage concerns -- every function operates over the
entire historical record available to it.

Cricket scoring conventions used here:
    - A batter's runs = sum of `batsman_runs` (does NOT include extras
      like wides/no-balls/byes, which are not credited to the batter).
    - Strike rate = (runs / balls faced) * 100. Balls faced excludes
      wides (a wide is not a legal delivery faced by the batter) but
      DOES include no-balls (the batter faces the ball and can score off
      it, even though it doesn't count as one of the bowler's six).
    - A bowler is credited with a wicket only for dismissal types they
      are directly responsible for: caught, caught and bowled, bowled,
      lbw, stumped, hit wicket. Run outs, retired hurt, and obstructing
      the field are NOT credited to the bowler.
    - Runs conceded by a bowler = `batsman_runs` + extras EXCEPT byes and
      leg-byes (byes/leg-byes are not the bowler's fault and don't count
      against their figures in standard cricket scoring).
    - Overs bowled = legal deliveries bowled / 6, displayed in cricket's
      "X.Y" notation (Y = balls 0-5 into the next over), NOT decimal
      division -- 1.5 overs means "1 over and 5 balls", not 1.83 overs.
    - Economy rate = runs conceded / (legal deliveries / 6).

Usage:
    from src.data.preprocess import preprocess_all
    from src.analytics.player_analytics import compute_batting_leaderboard

    deliveries_df = preprocess_all(save=False)["deliveries"]
    leaderboard = compute_batting_leaderboard(deliveries_df)
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd

from src.data.loader import DataValidationError
from src.utils.config import (
    BATSMAN_RUNS_COLUMN,
    BOWLER_COLUMN,
    DISMISSAL_TYPE_COLUMN,
    DISMISSED_PLAYER_COLUMN,
    EXTRA_TYPE_COLUMN,
    IS_WICKET_COLUMN,
    MATCH_ID_COLUMN,
    PLAYER_NAME_COLUMN,
    SEASON_COLUMN,
    STRIKER_COLUMN,
    TOTAL_RUNS_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "BOWLER_CREDITED_DISMISSALS",
    "MIN_BALLS_FOR_STRIKE_RATE_LEADERBOARD",
    "MIN_BALLS_FOR_ECONOMY_LEADERBOARD",
    "compute_batting_leaderboard",
    "compute_bowling_leaderboard",
    "compute_qualified_strike_rate_leaderboard",
    "compute_qualified_economy_leaderboard",
    "compute_player_profile",
    "compute_player_season_timeline",
    "compute_season_leaders",
]

# Dismissal types credited to the bowler. Anything else (run out, retired
# hurt, obstructing the field) is excluded from bowler wicket counts.
# A frozenset (not set) because this is a module-level constant that
# should never be mutated at runtime.
BOWLER_CREDITED_DISMISSALS: frozenset[str] = frozenset({
    "caught", "caught and bowled", "bowled", "lbw", "stumped", "hit wicket",
})

# Minimum legal balls faced/bowled for a player to appear on a *rate*
# leaderboard (strike rate, economy). Raw totals (runs, wickets) don't
# need a minimum -- but rates computed off a handful of balls are
# statistically meaningless and would otherwise dominate a "best strike
# rate" or "best economy" ranking. Used by `compute_qualified_*`.
MIN_BALLS_FOR_STRIKE_RATE_LEADERBOARD: int = 60
MIN_BALLS_FOR_ECONOMY_LEADERBOARD: int = 60

# Columns required on the input DataFrames for this module to function.
# Checked up front so a malformed/partial DataFrame fails fast with a
# clear message instead of raising a cryptic KeyError deep in a groupby.
_REQUIRED_DELIVERY_COLUMNS: frozenset[str] = frozenset({
    STRIKER_COLUMN, BOWLER_COLUMN, MATCH_ID_COLUMN, BATSMAN_RUNS_COLUMN,
    TOTAL_RUNS_COLUMN, EXTRA_TYPE_COLUMN, IS_WICKET_COLUMN,
    DISMISSAL_TYPE_COLUMN, DISMISSED_PLAYER_COLUMN,
})


class TeamRecordSplit(TypedDict):
    """Single-bucket record summary shared by a couple of dict-returning functions."""

    matches_played: int
    wins: int
    win_pct: float


class PlayerProfile(TypedDict):
    """Return shape of `compute_player_profile()`."""

    metadata: dict[str, Any] | None
    batting: dict[str, Any] | None
    bowling: dict[str, Any] | None


def _validate_deliveries_columns(deliveries_df: pd.DataFrame) -> None:
    """
    Verify that `deliveries_df` has the columns this module depends on.

    Args:
        deliveries_df: Deliveries DataFrame to validate.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = _REQUIRED_DELIVERY_COLUMNS - set(deliveries_df.columns)
    if missing:
        raise DataValidationError(
            f"deliveries DataFrame is missing required columns: {sorted(missing)}. "
            f"Did you pass the raw (uncleaned) deliveries.csv instead of the "
            f"output of `preprocess_all()`?"
        )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """
    Element-wise division that returns 0.0 wherever the denominator is 0,
    instead of raising or producing inf/NaN.

    Vectorized replacement for the previous `DataFrame.apply(axis=1, ...)`
    pattern, which re-invokes a Python lambda once per row and scales
    poorly past a few thousand players/rows.

    Args:
        numerator: Numerator values.
        denominator: Denominator values (zeros are handled safely).

    Returns:
        NumPy array of the same length as the inputs.
    """
    denom = denominator.to_numpy(dtype=float)
    safe_denom = np.where(denom == 0, 1.0, denom)
    result = numerator.to_numpy(dtype=float) / safe_denom
    return np.where(denom == 0, 0.0, result)


def compute_batting_leaderboard(deliveries_df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """
    Compute career batting statistics for every player who has faced at
    least one delivery, ranked by total runs scored.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        top_n: If given, return only the top N rows by total runs.

    Returns:
        DataFrame with columns: player, innings, runs, balls_faced, fours,
        sixes, strike_rate, dismissals, average. `average` is runs /
        dismissals (NaN-safe: shows total runs if the player has never
        been dismissed, matching standard cricket convention of an
        undefined/"not out" average). Empty (but correctly-columned) if
        `deliveries_df` is empty.

    Raises:
        DataValidationError: If `deliveries_df` is missing required columns.
    """
    _validate_deliveries_columns(deliveries_df)
    columns = [
        "player", "innings", "runs", "balls_faced", "fours", "sixes",
        "strike_rate", "dismissals", "average",
    ]
    if deliveries_df.empty:
        logger.warning("compute_batting_leaderboard() called with an empty deliveries DataFrame.")
        return pd.DataFrame(columns=columns)

    legal_balls = deliveries_df[deliveries_df[EXTRA_TYPE_COLUMN] != "wide"]

    runs = deliveries_df.groupby(STRIKER_COLUMN)[BATSMAN_RUNS_COLUMN].sum().rename("runs")
    balls_faced = legal_balls.groupby(STRIKER_COLUMN).size().rename("balls_faced")
    fours = (
        deliveries_df[deliveries_df[BATSMAN_RUNS_COLUMN] == 4]
        .groupby(STRIKER_COLUMN).size().rename("fours")
    )
    sixes = (
        deliveries_df[deliveries_df[BATSMAN_RUNS_COLUMN] == 6]
        .groupby(STRIKER_COLUMN).size().rename("sixes")
    )
    innings_count = deliveries_df.groupby(STRIKER_COLUMN)[MATCH_ID_COLUMN].nunique().rename("innings")
    dismissals = (
        deliveries_df[deliveries_df[DISMISSED_PLAYER_COLUMN].notna()]
        .groupby(DISMISSED_PLAYER_COLUMN)
        .size()
        .rename("dismissals")
    )

    leaderboard = pd.concat(
        [runs, balls_faced, fours, sixes, innings_count, dismissals], axis=1
    ).fillna(0)
    leaderboard.index.name = "player"
    leaderboard = leaderboard.reset_index()

    for col in ("runs", "balls_faced", "fours", "sixes", "innings", "dismissals"):
        leaderboard[col] = leaderboard[col].astype(int)

    leaderboard["strike_rate"] = np.round(
        _safe_divide(leaderboard["runs"], leaderboard["balls_faced"]) * 100, 2
    )
    # "Average" follows the cricket convention of being undefined for a
    # batter never dismissed; we fall back to showing total runs (a
    # common dashboard convention signalling "not out across the board"
    # rather than a true average), so we deliberately do NOT use
    # `_safe_divide` here, which would return 0.0 instead.
    leaderboard["average"] = np.where(
        leaderboard["dismissals"] > 0,
        np.round(leaderboard["runs"] / leaderboard["dismissals"].replace(0, np.nan), 2),
        leaderboard["runs"].astype(float),
    )

    leaderboard = leaderboard.sort_values("runs", ascending=False).reset_index(drop=True)
    if top_n is not None:
        leaderboard = leaderboard.head(top_n)
    logger.info(f"Computed batting leaderboard for {len(leaderboard)} players.")
    return leaderboard[columns]


def compute_bowling_leaderboard(deliveries_df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """
    Compute career bowling statistics for every player who has bowled at
    least one delivery, ranked by total wickets taken.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        top_n: If given, return only the top N rows by total wickets.

    Returns:
        DataFrame with columns: player, innings, overs, runs_conceded,
        wickets, economy, average, sorted by wickets descending (ties
        broken by economy ascending -- fewer runs per over among equally
        prolific wicket-takers ranks higher). Empty (but correctly-columned)
        if `deliveries_df` is empty.

    Raises:
        DataValidationError: If `deliveries_df` is missing required columns.
    """
    _validate_deliveries_columns(deliveries_df)
    columns = ["player", "innings", "overs", "runs_conceded", "wickets", "economy", "average"]
    if deliveries_df.empty:
        logger.warning("compute_bowling_leaderboard() called with an empty deliveries DataFrame.")
        return pd.DataFrame(columns=columns)

    legal_balls = deliveries_df[~deliveries_df[EXTRA_TYPE_COLUMN].isin(["wide", "no-ball"])]
    conceding_balls = deliveries_df[~deliveries_df[EXTRA_TYPE_COLUMN].isin(["bye", "leg-bye"])]

    legal_ball_count = legal_balls.groupby(BOWLER_COLUMN).size().rename("legal_balls")
    runs_conceded = conceding_balls.groupby(BOWLER_COLUMN)[TOTAL_RUNS_COLUMN].sum().rename("runs_conceded")
    innings_count = deliveries_df.groupby(BOWLER_COLUMN)[MATCH_ID_COLUMN].nunique().rename("innings")

    credited_wickets = deliveries_df[
        deliveries_df[IS_WICKET_COLUMN]
        & deliveries_df[DISMISSAL_TYPE_COLUMN].isin(BOWLER_CREDITED_DISMISSALS)
    ]
    wickets = credited_wickets.groupby(BOWLER_COLUMN).size().rename("wickets")

    leaderboard = pd.concat(
        [legal_ball_count, runs_conceded, innings_count, wickets], axis=1
    ).fillna(0)
    leaderboard.index.name = "player"
    leaderboard = leaderboard.reset_index()

    for col in ("legal_balls", "runs_conceded", "innings", "wickets"):
        leaderboard[col] = leaderboard[col].astype(int)

    # Cricket "X.Y overs" display notation (Y = balls into the next over,
    # 0-5), NOT true decimal division -- e.g. 7 legal balls -> "1.1", not
    # "1.17". This is intentionally a display value, not used in further
    # arithmetic (economy below recomputes from `legal_balls` directly).
    leaderboard["overs"] = (leaderboard["legal_balls"] // 6) + (leaderboard["legal_balls"] % 6) / 10

    overs_decimal = leaderboard["legal_balls"] / 6
    leaderboard["economy"] = np.round(_safe_divide(leaderboard["runs_conceded"], overs_decimal), 2)
    leaderboard["average"] = np.where(
        leaderboard["wickets"] > 0,
        np.round(leaderboard["runs_conceded"] / leaderboard["wickets"].replace(0, np.nan), 2),
        np.nan,
    )

    leaderboard = leaderboard.sort_values(
        ["wickets", "economy"], ascending=[False, True]
    ).reset_index(drop=True)
    if top_n is not None:
        leaderboard = leaderboard.head(top_n)
    logger.info(f"Computed bowling leaderboard for {len(leaderboard)} players.")
    return leaderboard[columns]


def compute_qualified_strike_rate_leaderboard(
    deliveries_df: pd.DataFrame,
    min_balls: int = MIN_BALLS_FOR_STRIKE_RATE_LEADERBOARD,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Rank players by strike rate, restricted to those who have faced at
    least `min_balls` deliveries.

    Strike rate is a *rate* statistic, so without a minimum-sample
    qualifier a player who has faced only a handful of balls (e.g. 3
    balls for 12 runs = 400 SR) can dominate a naive ranking despite an
    essentially meaningless sample size. This mirrors the qualification
    rule real-world cricket statisticians apply (analogous to ICC
    qualification thresholds for batting/bowling rankings).

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        min_balls: Minimum legal balls faced to qualify. Defaults to
            `MIN_BALLS_FOR_STRIKE_RATE_LEADERBOARD`.
        top_n: If given, return only the top N qualified players.

    Returns:
        DataFrame with the same columns as `compute_batting_leaderboard()`,
        filtered to qualified players and sorted by strike_rate descending.
    """
    leaderboard = compute_batting_leaderboard(deliveries_df)
    qualified = leaderboard[leaderboard["balls_faced"] >= min_balls]
    qualified = qualified.sort_values("strike_rate", ascending=False).reset_index(drop=True)
    logger.info(
        f"{len(qualified)}/{len(leaderboard)} players qualify for the strike-rate "
        f"leaderboard (min {min_balls} balls faced)."
    )
    return qualified.head(top_n) if top_n is not None else qualified


def compute_qualified_economy_leaderboard(
    deliveries_df: pd.DataFrame,
    min_balls: int = MIN_BALLS_FOR_ECONOMY_LEADERBOARD,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Rank bowlers by economy rate (ascending -- lower is better), restricted
    to those who have bowled at least `min_balls` legal deliveries.

    See `compute_qualified_strike_rate_leaderboard()` for why a minimum
    sample qualifier matters for rate statistics.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        min_balls: Minimum legal balls bowled to qualify. Defaults to
            `MIN_BALLS_FOR_ECONOMY_LEADERBOARD`.
        top_n: If given, return only the top N qualified bowlers.

    Returns:
        DataFrame with the same columns as `compute_bowling_leaderboard()`,
        filtered to qualified bowlers and sorted by economy ascending.
    """
    leaderboard = compute_bowling_leaderboard(deliveries_df)
    # Recover legal_balls from the display "overs" column would lose
    # precision, so recompute the qualifying ball count directly instead.
    legal_balls = (leaderboard["overs"].astype(int) * 6) + np.round(
        (leaderboard["overs"] - leaderboard["overs"].astype(int)) * 10
    ).astype(int)
    qualified = leaderboard[legal_balls >= min_balls]
    qualified = qualified.sort_values("economy", ascending=True).reset_index(drop=True)
    logger.info(
        f"{len(qualified)}/{len(leaderboard)} bowlers qualify for the economy "
        f"leaderboard (min {min_balls} legal balls bowled)."
    )
    return qualified.head(top_n) if top_n is not None else qualified


def compute_player_profile(
    deliveries_df: pd.DataFrame,
    players_df: pd.DataFrame,
    player_name: str,
    batting_leaderboard: pd.DataFrame | None = None,
    bowling_leaderboard: pd.DataFrame | None = None,
) -> PlayerProfile:
    """
    Build a single player's full profile: metadata from players.csv plus
    career batting and bowling figures.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        players_df: Cleaned players DataFrame.
        player_name: Exact player name to look up.
        batting_leaderboard: Optionally, a leaderboard already computed
            via `compute_batting_leaderboard(deliveries_df)`. Pass this in
            when looking up many players' profiles in a loop (e.g.
            rendering a full squad page) to avoid recomputing the
            league-wide leaderboard -- an O(n) scan -- on every call.
        bowling_leaderboard: Same idea for `compute_bowling_leaderboard()`.

    Returns:
        Dict with keys "metadata" (dict or None if not found in
        players.csv), "batting" (single-row dict from the batting
        leaderboard, or None if the player never batted), "bowling"
        (single-row dict from the bowling leaderboard, or None if the
        player never bowled).
    """
    metadata_rows = players_df[players_df[PLAYER_NAME_COLUMN] == player_name]
    metadata = metadata_rows.iloc[0].to_dict() if not metadata_rows.empty else None
    if metadata is None:
        logger.warning(f"No players.csv metadata found for player '{player_name}'.")

    batting_lb = (
        batting_leaderboard if batting_leaderboard is not None
        else compute_batting_leaderboard(deliveries_df)
    )
    batting_row = batting_lb[batting_lb["player"] == player_name]
    batting = batting_row.iloc[0].to_dict() if not batting_row.empty else None

    bowling_lb = (
        bowling_leaderboard if bowling_leaderboard is not None
        else compute_bowling_leaderboard(deliveries_df)
    )
    bowling_row = bowling_lb[bowling_lb["player"] == player_name]
    bowling = bowling_row.iloc[0].to_dict() if not bowling_row.empty else None

    if batting is None and bowling is None:
        logger.warning(f"Player '{player_name}' has no batting or bowling record in deliveries data.")

    return {"metadata": metadata, "batting": batting, "bowling": bowling}


def compute_player_season_timeline(
    deliveries_df: pd.DataFrame, matches_df: pd.DataFrame, player_name: str
) -> pd.DataFrame:
    """
    Compute a single player's runs scored and wickets taken in each IPL
    season they appeared in -- the player-level analogue of
    `team_analytics.compute_season_wise_performance()`, useful for a
    career trend line on the Player Analytics dashboard page.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        matches_df: Cleaned matches DataFrame (used to map match_id -> season).
        player_name: Exact player name to look up.

    Returns:
        DataFrame with columns: season, runs, wickets, sorted by season
        ascending. Seasons in which the player neither batted nor bowled
        a single delivery are simply absent (not zero-filled), since
        "didn't play that season" and "played and scored zero" are
        different facts a dashboard may want to distinguish.
    """
    _validate_deliveries_columns(deliveries_df)
    match_to_season = matches_df.set_index(MATCH_ID_COLUMN)[SEASON_COLUMN]
    deliveries = deliveries_df.copy()
    deliveries["season"] = deliveries[MATCH_ID_COLUMN].map(match_to_season)

    batting = deliveries[deliveries[STRIKER_COLUMN] == player_name]
    runs_by_season = batting.groupby("season")[BATSMAN_RUNS_COLUMN].sum().rename("runs")

    bowling = deliveries[
        (deliveries[BOWLER_COLUMN] == player_name)
        & deliveries[IS_WICKET_COLUMN]
        & deliveries[DISMISSAL_TYPE_COLUMN].isin(BOWLER_CREDITED_DISMISSALS)
    ]
    wickets_by_season = bowling.groupby("season").size().rename("wickets")

    timeline = pd.concat([runs_by_season, wickets_by_season], axis=1).fillna(0).reset_index()
    timeline = timeline.rename(columns={"index": "season"})
    timeline["runs"] = timeline["runs"].astype(int)
    timeline["wickets"] = timeline["wickets"].astype(int)
    timeline = timeline.sort_values("season").reset_index(drop=True)

    if timeline.empty:
        logger.warning(f"No season timeline data found for player '{player_name}'.")

    return timeline[["season", "runs", "wickets"]]


def compute_season_leaders(
    deliveries_df: pd.DataFrame, matches_df: pd.DataFrame, season: int
) -> dict[str, pd.DataFrame]:
    """
    Compute the orange cap (most runs) and purple cap (most wickets)
    contenders for a single season, ranked, for cross-referencing against
    `seasons.csv`'s official `orange_cap_winner` / `purple_cap_winner`.

    Args:
        deliveries_df: Cleaned deliveries DataFrame.
        matches_df: Cleaned matches DataFrame (used to map match_id -> season).
        season: Season year to filter to.

    Returns:
        Dict with keys "top_batters" and "top_bowlers", each a top-10
        leaderboard DataFrame restricted to deliveries from that season.
        Both are empty (but correctly-columned) if no matches are found
        for the given season.
    """
    season_match_ids = set(matches_df[matches_df[SEASON_COLUMN] == season][MATCH_ID_COLUMN])
    if not season_match_ids:
        logger.warning(f"No matches found for season {season} in matches_df.")

    season_deliveries = deliveries_df[deliveries_df[MATCH_ID_COLUMN].isin(season_match_ids)]
    logger.info(f"Computing season leaders for {season}: {len(season_deliveries):,} deliveries.")

    return {
        "top_batters": compute_batting_leaderboard(season_deliveries, top_n=10),
        "top_bowlers": compute_bowling_leaderboard(season_deliveries, top_n=10),
    }
