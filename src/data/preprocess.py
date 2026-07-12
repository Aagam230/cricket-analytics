"""
preprocess.py
=============
Data cleaning layer for the Cricket Analytics & Match Prediction Platform.

This module takes the RAW DataFrames produced by `src/data/loader.py` and
turns them into the CANONICAL, cleaned DataFrames every other module in
this project consumes -- matching the column names defined in
`src/utils/config.py`. It stops short of building model-ready features
(that's `src/features/engineering.py`'s job).

------------------------------------------------------------------------
2026-07-01 DATASET MIGRATION NOTE -- READ THIS FIRST
------------------------------------------------------------------------
This is the file that absorbs almost the entire schema migration. The new
dataset's raw column names/files are quite different from the old
synthetic dataset (see `src/data/loader.py` for the exact raw headers),
but every module downstream of this one (`src/features/engineering.py`,
`src/analytics/*`, `src/visualization/plots.py`) still expects the SAME
canonical columns it always has (`match_id`, `winner`, `striker`,
`total_runs`, `first_innings_score`, ...). That is intentional: by doing
all the raw-to-canonical translation HERE, none of those other modules
needed to change at all.

Concretely, this module now does several things the old version didn't
need to:

    1. RENAME raw columns to their canonical equivalents (e.g. `matchId`
       -> `match_id`, `batsman` -> `striker`, `inning` -> `innings`).
    2. DERIVE columns that don't exist in the raw files at all:
        - `total_runs` = batsman_runs + extras (deliveries)
        - `extra_type` = which of isWide/isNoBall/Byes/LegByes/Penalty is
          set on a given ball (deliveries)
        - `is_wicket` = dismissal_kind is not null (deliveries)
        - `first_innings_score` / `first_innings_wickets` /
          `first_innings_overs` and the `second_innings_*` equivalents,
          aggregated from ball-by-ball data per match (matches)
        - `win_by` / `win_margin`, derived from `winner_runs` /
          `winner_wickets` (matches)
        - `result`, derived from the raw `outcome` column (matches)
        - `match_number`, re-derived from chronological order within a
          season because ~6% of the raw values are missing (matches)
    3. DERIVE two entire DataFrames -- `players_clean` and
       `seasons_clean` -- that have NO raw-file source anymore. See
       `derive_players()` and `derive_seasons()` below.
    4. DROP one feature with no replacement: `is_day_night`. There is no
       day/night or match-start-time field anywhere in the new dataset,
       so it is simply absent from `matches_clean.csv` going forward (see
       the migration note in `src/utils/config.py`).

Responsibilities preserved from the original design:
    1. Normalize team names across all relevant columns (so "Delhi
       Daredevils" and "Delhi Capitals" -- or, new in this dataset,
       "Royal Challengers Bangalore" and "Royal Challengers Bengaluru" --
       are treated as the same team).
    2. Handle matches with no `winner` (ties / no-results) -- kept for
       analytics, excluded from anything used as an ML training target.
    3. Derive simple date-based columns (year, month, day_of_week) from
       the `date` column for trend analysis.
    4. Run sanity-check validations and log anything suspicious rather
       than silently proceeding.
    5. Persist cleaned DataFrames to data/processed/ so downstream steps
       never touch raw data directly.

Usage:
    from src.data.preprocess import preprocess_all

    cleaned = preprocess_all()
    matches_clean = cleaned["matches"]

Or from the command line:
    python -m src.data.preprocess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import load_all
from src.utils.config import (
    PROCESSED_DELIVERIES_PATH,
    PROCESSED_MATCHES_PATH,
    PROCESSED_PLAYERS_PATH,
    PROCESSED_SEASONS_PATH,
    TEAM_NAME_MAPPING,
    ensure_directories_exist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns across the cleaned datasets that contain team names and
# therefore need TEAM_NAME_MAPPING applied. Centralized here so adding a
# new dataset later only requires extending this list, not hunting
# through the file.
MATCHES_TEAM_COLUMNS: list[str] = ["team1", "team2", "toss_winner", "winner"]
DELIVERIES_TEAM_COLUMNS: list[str] = ["batting_team", "bowling_team"]
SEASONS_TEAM_COLUMNS: list[str] = ["champion", "runner_up"]

# Dismissal types credited to the BOWLER (mirrors
# `src.analytics.player_analytics.BOWLER_CREDITED_DISMISSALS` -- kept as
# its own local copy rather than imported, so the data-layer
# (`src/data/`) doesn't depend on the analytics layer (`src/analytics/`),
# preserving the project's existing layering).
_BOWLER_CREDITED_DISMISSALS: frozenset[str] = frozenset({
    "caught", "caught and bowled", "bowled", "lbw", "stumped", "hit wicket",
})

# A small number of historical neutral-venue matches (in the UAE) have a
# null `city` in the raw data. These are the only two venues affected
# (verified 2026-07-01); everything else has a city.
_VENUE_CITY_FALLBACK: dict[str, str] = {
    "Sharjah Cricket Stadium": "Sharjah",
    "Dubai International Cricket Stadium": "Dubai",
}


# ====================================================================
# Shared helpers
# ====================================================================

def normalize_team_names(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply TEAM_NAME_MAPPING to standardize historical team names to their
    current equivalents across the given columns.

    Args:
        df: DataFrame containing one or more team-name columns.
        columns: Names of columns to normalize. Columns not present in the
            DataFrame are silently skipped (so this function can be reused
            across matches/deliveries/seasons without per-dataset branching).

    Returns:
        A new DataFrame (the original is not mutated) with team names
        replaced according to TEAM_NAME_MAPPING. Names not present in the
        mapping are left unchanged.

    Example:
        >>> df = pd.DataFrame({"team1": ["Delhi Daredevils"]})
        >>> normalize_team_names(df, ["team1"])["team1"].iloc[0]
        'Delhi Capitals'
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        n_changed = df[col].isin(TEAM_NAME_MAPPING.keys()).sum()
        if n_changed > 0:
            df[col] = df[col].replace(TEAM_NAME_MAPPING)
            logger.info(f"Normalized {n_changed} team name(s) in column '{col}'")
    return df


def _normalize_season(raw_season: pd.Series) -> pd.Series:
    """
    Normalize the raw `season` column to a single comparable int.

    The raw data mixes single-year seasons ("2009", "2011", ...) with
    split-year seasons ("2007/08", "2009/10", "2020/21") for years the
    tournament straddled a calendar-year boundary. This collapses every
    season to the year it STARTED in (e.g. "2007/08" -> 2007), matching
    the plain-int convention the rest of the pipeline (and the old
    dataset) used for `season`.

    Args:
        raw_season: Raw `season` column (strings like "2020/21" or "2019").

    Returns:
        Int Series, one value per row.
    """
    return raw_season.astype(str).str.slice(0, 4).astype(int)


def _derive_extra_type(deliveries_raw: pd.DataFrame) -> pd.Series:
    """
    Collapse the raw dataset's five separate extra-type flag/amount
    columns (isWide, isNoBall, Byes, LegByes, Penalty -- each holding the
    run amount when that extra type applies, NaN otherwise) into the
    single `extra_type` category column the rest of the pipeline expects
    (matching the old dataset's `extra_type` semantics: 'wide',
    'no-ball', 'bye', 'leg-bye', or NaN for a clean delivery).

    A small number of balls (28 out of ~278k, e.g. a no-ball followed by
    byes) have MORE THAN ONE flag set. Real cricket scoring would record
    both, but the old (and downstream) schema only has room for a single
    `extra_type` per ball, so this picks one via the priority order
    below -- chosen so the the legality-affecting extras (wide/no-ball)
    always win over the byes/leg-byes/penalty categories that don't
    affect whether the ball counts toward the bowler's over. This is a
    deliberate, documented approximation affecting <0.01% of deliveries.

    Adds a new category, 'penalty', that didn't exist in the old
    dataset's `extra_type` values -- the new dataset records penalty
    runs explicitly, so they're preserved here rather than discarded.

    Args:
        deliveries_raw: Raw deliveries DataFrame (post `load_deliveries()`).

    Returns:
        String Series (object dtype, NaN where no extra applies).
    """
    conditions = [
        deliveries_raw["isWide"].notna(),
        deliveries_raw["isNoBall"].notna(),
        deliveries_raw["Penalty"].notna(),
        deliveries_raw["Byes"].notna(),
        deliveries_raw["LegByes"].notna(),
    ]
    choices = ["wide", "no-ball", "penalty", "bye", "leg-bye"]
    result = np.select(conditions, choices, default="")
    return pd.Series(result, index=deliveries_raw.index).replace("", np.nan)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive year, month, and day_of_week columns from the `date` column.

    Args:
        df: matches DataFrame with a `date` column already parsed as
            datetime (this is done in loader.py's load_matches()).

    Returns:
        A new DataFrame with three additional columns:
            - `match_year`: calendar year of the match
            - `match_month`: calendar month (1-12)
            - `match_day_of_week`: day name, e.g. "Monday"
        These support seasonal/temporal trend analysis in the Match
        Insights dashboard (e.g. "are weekend matches higher scoring?").
    """
    df = df.copy()
    df["match_year"] = df["date"].dt.year
    df["match_month"] = df["date"].dt.month
    df["match_day_of_week"] = df["date"].dt.day_name()
    return df


def flag_no_result_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an explicit boolean flag for matches with no winner (ties or
    no-results), rather than silently dropping them.

    Why this matters:
        Dropping rows with missing data is sometimes the right call, but
        doing it silently hides information. A tie/no-result match still
        happened -- it should count in venue statistics, season match
        counts, and team participation totals. It should NOT be used as a
        training example for the win-prediction model, since there's no
        valid "winner" label to learn from. This function makes that
        distinction explicit and queryable rather than baking the decision
        into a silent .dropna() call buried in the modeling code.

    Args:
        df: matches DataFrame with a `winner` column.

    Returns:
        A new DataFrame with an added boolean column `is_no_result`,
        True where `winner` is null.
    """
    df = df.copy()
    df["is_no_result"] = df["winner"].isna()
    n_no_result = df["is_no_result"].sum()
    logger.info(
        f"Flagged {n_no_result} no-result/tied matches via 'is_no_result' "
        f"column (kept in dataset for analytics, must be excluded by "
        f"feature engineering when building the ML training set)."
    )
    return df


def validate_matches(df: pd.DataFrame) -> None:
    """
    Run sanity checks on cleaned matches data and log warnings for
    anything suspicious. Does not raise or drop rows -- validation here is
    about visibility, not enforcement, since some "issues" (e.g. a
    genuinely tied match) are valid data, not bugs.

    Checks performed:
        1. No duplicate match_id values.
        2. toss_winner is always one of team1/team2 (for matches with
           non-null teams).
        3. winner (where present) is always one of team1/team2.

    Args:
        df: Cleaned matches DataFrame.
    """
    n_duplicates = df["match_id"].duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate match_id values in matches data.")

    invalid_toss_mask = ~df.apply(
        lambda row: row["toss_winner"] in (row["team1"], row["team2"]), axis=1
    )
    n_bad_toss = invalid_toss_mask.sum()
    if n_bad_toss > 0:
        logger.warning(
            f"Found {n_bad_toss} matches where toss_winner is not team1 or "
            f"team2 -- check for team name normalization gaps."
        )

    valid_winner_rows = df[~df["is_no_result"]]
    bad_winner_mask = ~valid_winner_rows.apply(
        lambda row: row["winner"] in (row["team1"], row["team2"]), axis=1
    )
    n_bad_winner = bad_winner_mask.sum()
    if n_bad_winner > 0:
        logger.warning(
            f"Found {n_bad_winner} matches where winner is not team1 or "
            f"team2 -- check for team name normalization gaps."
        )

    if n_duplicates == 0 and n_bad_toss == 0 and n_bad_winner == 0:
        logger.info("matches.csv validation passed: no duplicates, toss/winner consistency OK.")


# ====================================================================
# deliveries.csv -- raw to canonical
# ====================================================================

def preprocess_deliveries(deliveries_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for deliveries: rename/derive raw columns into
    the canonical schema, then normalize team names.

    Raw -> canonical column mapping:
        matchId      -> match_id
        inning       -> innings
        over (0-19)  -> over (1-20, re-indexed to cricket's 1-based
                        convention to match the old dataset; the raw data
                        is 0-indexed)
        ball         -> ball                       (unchanged)
        batting_team -> batting_team                (unchanged)
        bowling_team -> bowling_team                (unchanged)
        batsman      -> striker
        non_striker  -> non_striker                 (unchanged)
        bowler       -> bowler                      (unchanged)
        batsman_runs -> batsman_runs                (unchanged)
        extras       -> extra_runs
        (derived)    -> total_runs = batsman_runs + extras
        (derived)    -> extra_type (see `_derive_extra_type()`)
        dismissal_kind     -> dismissal_type
        player_dismissed   -> dismissed_player       (unchanged)
        (derived)    -> is_wicket = dismissal_type.notna()
        (no source)  -> fielder = NaN (the new dataset records no
                        fielder/catcher name anywhere -- closest
                        meaningful alternative is `dismissed_player` +
                        `dismissal_type`, which ARE preserved; "who took
                        the catch" simply isn't recorded in this dataset)
        (derived)    -> delivery_id = sequential "D0000001", "D0000002", ...

    Args:
        deliveries_raw: Raw deliveries DataFrame from load_deliveries().

    Returns:
        Cleaned, canonically-named deliveries DataFrame, ready for
        feature engineering / analytics.
    """
    logger.info("Preprocessing deliveries.csv...")
    df = deliveries_raw.copy()

    extra_type = _derive_extra_type(df)
    total_runs = df["batsman_runs"].fillna(0) + df["extras"].fillna(0)

    clean = pd.DataFrame({
        "match_id": df["matchId"],
        "innings": df["inning"],
        "over": df["over"] + 1,  # raw is 0-indexed; old schema/cricket convention is 1-indexed
        "ball": df["ball"],
        "batting_team": df["batting_team"],
        "bowling_team": df["bowling_team"],
        "striker": df["batsman"],
        "non_striker": df["non_striker"],
        "bowler": df["bowler"],
        "batsman_runs": df["batsman_runs"],
        "extra_runs": df["extras"],
        "total_runs": total_runs.astype(int),
        "extra_type": extra_type,
        "is_wicket": df["dismissal_kind"].notna(),
        "dismissal_type": df["dismissal_kind"],
        "dismissed_player": df["player_dismissed"],
        "fielder": np.nan,  # not available in this dataset -- see docstring
    })
    clean.insert(0, "delivery_id", [f"D{i + 1:07d}" for i in range(len(clean))])

    clean = normalize_team_names(clean, DELIVERIES_TEAM_COLUMNS)
    logger.info(f"deliveries.csv preprocessing complete: {clean.shape[0]:,} rows retained.")
    return clean


# ====================================================================
# matches.csv -- raw to canonical (depends on cleaned deliveries for
# the innings-score aggregates, since the new dataset doesn't carry
# innings totals on the matches file the way the old one did)
# ====================================================================

def _compute_innings_aggregates(deliveries_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data into per-match first/second innings
    totals -- the new dataset has NO innings-score columns on the
    matches file at all (the old dataset's `first_innings_score`,
    `first_innings_wickets`, `first_innings_overs`, and the
    `second_innings_*` equivalents simply don't exist in the raw new
    matches.csv). This is the closest meaningful alternative: derive the
    exact same statistics directly from the ball-by-ball record, which
    is strictly more accurate than a pre-aggregated column would be.

    Innings 3+ (super overs) are intentionally excluded -- "first/second
    innings" here means the two main innings of regulation play.

    Args:
        deliveries_clean: Output of `preprocess_deliveries()`.

    Returns:
        DataFrame with one row per `match_id` and columns
        `first_innings_score`, `first_innings_wickets`,
        `first_innings_overs`, `second_innings_score`,
        `second_innings_wickets`, `second_innings_overs`. Matches with no
        ball-by-ball record for a given innings (e.g. abandoned before a
        ball was bowled) simply have NaN in that innings' columns.
    """
    df = deliveries_clean[deliveries_clean["innings"].isin([1, 2])].copy()
    is_legal = ~df["extra_type"].isin(["wide", "no-ball"])

    grouped = (
        df.assign(_legal=is_legal)
        .groupby(["match_id", "innings"])
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("_legal", "sum"),
        )
        .reset_index()
    )
    # Cricket "X.Y overs" display notation (Y = balls into the next over,
    # 0-5), matching the convention already used in
    # `analytics/player_analytics.py`'s bowling leaderboard.
    grouped["overs"] = (grouped["legal_balls"] // 6) + (grouped["legal_balls"] % 6) / 10

    first = (
        grouped[grouped["innings"] == 1]
        .set_index("match_id")[["runs", "wickets", "overs"]]
        .rename(columns={"runs": "first_innings_score", "wickets": "first_innings_wickets", "overs": "first_innings_overs"})
    )
    second = (
        grouped[grouped["innings"] == 2]
        .set_index("match_id")[["runs", "wickets", "overs"]]
        .rename(columns={"runs": "second_innings_score", "wickets": "second_innings_wickets", "overs": "second_innings_overs"})
    )
    return first.join(second, how="outer").reset_index()


def preprocess_matches(matches_raw: pd.DataFrame, deliveries_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for matches: rename/derive raw columns into
    the canonical schema (merging in ball-by-ball innings aggregates),
    normalize team names, flag no-result matches, add date features, and
    validate.

    Raw -> canonical column mapping/derivation:
        matchId            -> match_id
        season ("2020/21") -> season (int 2020, see `_normalize_season()`)
        match_number        -> match_number (RE-DERIVED from chronological
                               order within season -- ~6% of raw values
                               are missing; see docstring note below)
        (no source)         -> stage = NaN (the new dataset has no
                               tournament-phase/stage field -- `event`,
                               `eliminator`, and `method` exist but
                               describe different things, see below.
                               `stage` is unused anywhere downstream
                               except as a display column, so this is a
                               safe, explicit gap rather than a silent one)
        date                -> date                          (unchanged)
        venue               -> venue                         (unchanged)
        city (51 nulls)     -> city (nulls filled for the two known
                               neutral UAE venues; see
                               `_VENUE_CITY_FALLBACK`)
        team1 / team2       -> team1 / team2                 (unchanged)
        toss_winner         -> toss_winner                   (unchanged)
        toss_decision       -> toss_decision                 (unchanged)
        (derived, see        -> first_innings_score/wickets/overs,
         `_compute_innings_     second_innings_score/wickets/overs
         aggregates()`)
        outcome (nan/'tie'/  -> result ('normal'/'tie'/'no result')
         'no result')
        winner              -> winner                        (unchanged)
        winner_runs,         -> win_by ('runs'/'wickets'/NaN),
         winner_wickets        win_margin (the matching numeric value)
        player_of_match      -> player_of_match               (unchanged)
        umpire1 / umpire2    -> umpire1 / umpire2              (unchanged)
        (no source)          -> is_day_night DROPPED ENTIRELY -- no
                               day/night or start-time field exists
                               anywhere in this dataset (see the
                               migration note in `src/utils/config.py`)

        event, gender, eliminator, method, neutralvenue,
        reserve_umpire, tv_umpire, match_referee are carried through
        UNCHANGED as bonus columns -- they have no equivalent in the old
        schema, aren't required by any downstream module, but are kept
        since they're free extra context (e.g. `eliminator` records which
        team won a Super Over tie-breaker).

    Args:
        matches_raw: Raw matches DataFrame from load_matches().
        deliveries_clean: Output of `preprocess_deliveries()`, used to
            compute the innings-score aggregates.

    Returns:
        Cleaned, canonically-named matches DataFrame, ready for feature
        engineering / analytics.
    """
    logger.info("Preprocessing matches.csv...")
    df = matches_raw.copy()

    season = _normalize_season(df["season"])
    city = df["city"].fillna(df["venue"].map(_VENUE_CITY_FALLBACK))
    n_city_still_missing = city.isna().sum()
    if n_city_still_missing > 0:
        logger.warning(
            f"{n_city_still_missing} matches still have no city after applying "
            f"known neutral-venue fallbacks -- filling with 'Unknown'."
        )
        city = city.fillna("Unknown")

    outcome = df["outcome"]
    result = np.select(
        [outcome == "tie", outcome == "no result"],
        ["tie", "no result"],
        default="normal",
    )

    win_by = np.select(
        [df["winner_runs"].fillna(0) > 0, df["winner_wickets"].fillna(0) > 0],
        ["runs", "wickets"],
        default="",
    )
    win_by = pd.Series(win_by, index=df.index).replace("", np.nan)
    win_margin = np.where(
        win_by == "runs", df["winner_runs"],
        np.where(win_by == "wickets", df["winner_wickets"], np.nan),
    )

    clean = pd.DataFrame({
        "match_id": df["matchId"],
        "season": season,
        "match_number": np.nan,  # re-derived chronologically below
        "stage": np.nan,         # not available in this dataset -- see docstring
        "date": df["date"],
        "venue": df["venue"],
        "city": city,
        "team1": df["team1"],
        "team2": df["team2"],
        "toss_winner": df["toss_winner"],
        "toss_decision": df["toss_decision"],
        "result": result,
        "winner": df["winner"],
        "win_by": win_by,
        "win_margin": win_margin,
        "player_of_match": df["player_of_match"],
        "umpire1": df["umpire1"],
        "umpire2": df["umpire2"],
        # Bonus passthrough columns with no canonical equivalent:
        "event": df["event"],
        "gender": df["gender"],
        "eliminator": df["eliminator"],
        "method": df["method"],
        "neutral_venue": df["neutralvenue"],
        "reserve_umpire": df["reserve_umpire"],
        "tv_umpire": df["tv_umpire"],
        "match_referee": df["match_referee"],
    })

    # match_number: ~6% of raw values are missing, so rather than leaving
    # holes, re-derive a complete, consistent sequence from chronological
    # order within each season. This is a documented approximation -- it
    # assumes matches were played in date order within a season, which
    # holds for IPL's round-robin + playoffs structure.
    clean = clean.sort_values(["season", "date", "match_id"]).reset_index(drop=True)
    clean["match_number"] = clean.groupby("season").cumcount() + 1

    # Merge in ball-by-ball innings aggregates (first/second innings
    # score, wickets, overs) -- see `_compute_innings_aggregates()`.
    innings_agg = _compute_innings_aggregates(deliveries_clean)
    clean = clean.merge(innings_agg, on="match_id", how="left")
    n_missing_innings = clean["first_innings_score"].isna().sum()
    if n_missing_innings > 0:
        logger.warning(
            f"{n_missing_innings} matches have no ball-by-ball record for "
            f"the first innings (likely abandoned before a ball was "
            f"bowled) -- first/second innings columns are NaN for these."
        )

    clean = normalize_team_names(clean, MATCHES_TEAM_COLUMNS)
    clean = flag_no_result_matches(clean)
    clean = add_date_features(clean)
    validate_matches(clean)
    logger.info(f"matches.csv preprocessing complete: {clean.shape[0]:,} rows retained.")
    return clean


# ====================================================================
# players_clean.csv -- DERIVED (no raw source file in this dataset)
# ====================================================================

def derive_players(deliveries_clean: pd.DataFrame, matches_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a player roster DataFrame from ball-by-ball appearances, since
    this dataset ships no players.csv at all.

    This is the closest meaningful alternative to the old players.csv:
    every player's name, debut season, last season played, and basic
    batting/bowling participation can be reconstructed directly from the
    deliveries data. What CANNOT be recovered -- because the information
    simply doesn't exist anywhere in this dataset -- is: nationality,
    date of birth, batting/bowling style, playing role, capped/uncapped
    international status, and auction prices. Those columns are not
    fabricated; they're absent from the derived schema entirely (see
    `src/utils/config.py` for the column constants that were removed).

    Args:
        deliveries_clean: Output of `preprocess_deliveries()`.
        matches_clean: Output of `preprocess_matches()` (used to map
            match_id -> season).

    Returns:
        DataFrame with columns: player_id, player_name, ipl_debut_season,
        last_season_played, matches_batted, matches_bowled, career_runs,
        career_wickets -- one row per distinct player name seen as a
        striker, non-striker, or bowler. Sorted alphabetically by name.
    """
    logger.info("Deriving players.csv (no raw source file in this dataset)...")
    match_to_season = matches_clean.set_index("match_id")["season"]
    deliveries = deliveries_clean.copy()
    deliveries["season"] = deliveries["match_id"].map(match_to_season)

    batting = deliveries.groupby("striker").agg(
        bat_first_season=("season", "min"),
        bat_last_season=("season", "max"),
        matches_batted=("match_id", "nunique"),
        career_runs=("batsman_runs", "sum"),
    )
    bowling = deliveries.groupby("bowler").agg(
        bowl_first_season=("season", "min"),
        bowl_last_season=("season", "max"),
        matches_bowled=("match_id", "nunique"),
    )
    credited_wickets = deliveries[
        deliveries["is_wicket"] & deliveries["dismissal_type"].isin(_BOWLER_CREDITED_DISMISSALS)
    ]
    wickets = credited_wickets.groupby("bowler").size().rename("career_wickets")

    roster = batting.join(bowling, how="outer").join(wickets, how="outer")
    roster.index.name = "player_name"
    roster = roster.reset_index()

    roster["ipl_debut_season"] = roster[["bat_first_season", "bowl_first_season"]].min(axis=1)
    roster["last_season_played"] = roster[["bat_last_season", "bowl_last_season"]].max(axis=1)
    roster["matches_batted"] = roster["matches_batted"].fillna(0).astype(int)
    roster["matches_bowled"] = roster["matches_bowled"].fillna(0).astype(int)
    roster["career_runs"] = roster["career_runs"].fillna(0).astype(int)
    roster["career_wickets"] = roster["career_wickets"].fillna(0).astype(int)

    roster = roster.sort_values("player_name").reset_index(drop=True)
    roster.insert(0, "player_id", [f"P{i + 1:04d}" for i in range(len(roster))])

    players = roster[[
        "player_id", "player_name", "ipl_debut_season", "last_season_played",
        "matches_batted", "matches_bowled", "career_runs", "career_wickets",
    ]]
    logger.info(f"Derived players.csv: {len(players):,} distinct players.")
    return players


# ====================================================================
# seasons_clean.csv -- DERIVED (no raw source file in this dataset)
# ====================================================================

def derive_seasons(matches_clean: pd.DataFrame, deliveries_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a season-level summary DataFrame by aggregating the cleaned
    matches + deliveries data, since this dataset ships no seasons.csv.

    The champion/runner-up for each season is inferred as the winner/
    loser of that season's LAST match in chronological order (i.e. the
    final) -- a reasonable inference given IPL's round-robin + playoffs
    structure, but an inference nonetheless (the raw data has no explicit
    "this was the final" flag), so it's documented here rather than
    presented as ground truth. `most_valuable_player` from the old schema
    is dropped -- there's no reliable way to derive "most valuable"
    (distinct from top run-scorer/wicket-taker) from ball-by-ball data
    alone, and fabricating it would be misleading.

    Args:
        matches_clean: Output of `preprocess_matches()`.
        deliveries_clean: Output of `preprocess_deliveries()`.

    Returns:
        DataFrame with columns: season, total_matches, num_teams,
        champion, runner_up, orange_cap_winner, purple_cap_winner,
        total_runs_scored, total_sixes, total_fours,
        avg_first_innings_score, highest_team_total, lowest_team_total --
        one row per season, sorted ascending.
    """
    logger.info("Deriving seasons.csv (no raw source file in this dataset)...")

    # Import kept local to this function to avoid a module-level
    # data-layer -> analytics-layer import cycle; `compute_season_leaders`
    # is reused as-is rather than re-implemented, per the project's
    # "reuse existing code" principle.
    from src.analytics.player_analytics import compute_season_leaders

    rows = []
    for season, season_matches in matches_clean.groupby("season"):
        decided = season_matches[~season_matches["is_no_result"]]
        teams = pd.concat([season_matches["team1"], season_matches["team2"]])

        final_match = season_matches.sort_values(["date", "match_number"]).iloc[-1]
        if pd.isna(final_match["winner"]):
            champion, runner_up = np.nan, np.nan
            logger.warning(
                f"Season {season}: the chronologically last match was a "
                f"tie/no-result -- champion/runner_up could not be inferred."
            )
        else:
            champion = final_match["winner"]
            runner_up = final_match["team2"] if final_match["winner"] == final_match["team1"] else final_match["team1"]

        leaders = compute_season_leaders(deliveries_clean, matches_clean, season)
        orange_cap = leaders["top_batters"].iloc[0]["player"] if not leaders["top_batters"].empty else np.nan
        purple_cap = leaders["top_bowlers"].iloc[0]["player"] if not leaders["top_bowlers"].empty else np.nan

        season_match_ids = set(season_matches["match_id"])
        season_deliveries = deliveries_clean[deliveries_clean["match_id"].isin(season_match_ids)]

        innings_totals = pd.concat([
            season_matches["first_innings_score"], season_matches["second_innings_score"],
        ]).dropna()

        rows.append({
            "season": season,
            "total_matches": int(len(season_matches)),
            "num_teams": int(teams.nunique()),
            "champion": champion,
            "runner_up": runner_up,
            "orange_cap_winner": orange_cap,
            "purple_cap_winner": purple_cap,
            "total_runs_scored": int(season_deliveries["total_runs"].sum()),
            "total_sixes": int((season_deliveries["batsman_runs"] == 6).sum()),
            "total_fours": int((season_deliveries["batsman_runs"] == 4).sum()),
            "avg_first_innings_score": round(float(season_matches["first_innings_score"].mean()), 1),
            "highest_team_total": float(innings_totals.max()) if not innings_totals.empty else np.nan,
            "lowest_team_total": float(innings_totals.min()) if not innings_totals.empty else np.nan,
            "no_results": int(season_matches["is_no_result"].sum()),
        })

    seasons = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    seasons = normalize_team_names(seasons, SEASONS_TEAM_COLUMNS)
    logger.info(f"Derived seasons.csv: {len(seasons):,} seasons.")
    return seasons


# ====================================================================
# Top-level orchestration
# ====================================================================

def preprocess_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run the full preprocessing pipeline for all four logical datasets.

    Order matters here: deliveries must be cleaned BEFORE matches (so
    matches can merge in ball-by-ball innings aggregates), and both must
    be cleaned before players/seasons can be derived from them.

    Args:
        save: If True (default), write each cleaned/derived DataFrame to
            its configured path in data/processed/. Set to False if you
            only want the DataFrames in memory (e.g. for testing).

    Returns:
        Dictionary with keys "matches", "deliveries", "players", "seasons",
        each mapping to its cleaned/derived DataFrame -- the SAME shape of
        dict the original 4-raw-file pipeline returned, even though
        "players" and "seasons" no longer come from raw files.
    """
    ensure_directories_exist()
    raw = load_all()

    deliveries_clean = preprocess_deliveries(raw["deliveries"])
    matches_clean = preprocess_matches(raw["matches"], deliveries_clean)
    players_clean = derive_players(deliveries_clean, matches_clean)
    seasons_clean = derive_seasons(matches_clean, deliveries_clean)

    cleaned = {
        "matches": matches_clean,
        "deliveries": deliveries_clean,
        "players": players_clean,
        "seasons": seasons_clean,
    }

    if save:
        cleaned["matches"].to_csv(PROCESSED_MATCHES_PATH, index=False)
        cleaned["deliveries"].to_csv(PROCESSED_DELIVERIES_PATH, index=False)
        cleaned["players"].to_csv(PROCESSED_PLAYERS_PATH, index=False)
        cleaned["seasons"].to_csv(PROCESSED_SEASONS_PATH, index=False)
        logger.info(
            f"Saved cleaned datasets to {PROCESSED_MATCHES_PATH.parent}"
        )

    return cleaned


if __name__ == "__main__":
    # Allows running `python -m src.data.preprocess` directly to execute
    # the full cleaning pipeline and persist outputs to data/processed/.
    results = preprocess_all()
    for name, frame in results.items():
        print(f"{name}: {frame.shape}")
