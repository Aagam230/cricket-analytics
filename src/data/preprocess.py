"""
preprocess.py
=============
Data cleaning layer for the Cricket Analytics & Match Prediction Platform.

This module takes the raw DataFrames produced by `src/data/loader.py` and
applies cleaning, normalization, and validation -- but stops short of
building model-ready features (that's `src/features/engineering.py`'s job).

Responsibilities of this module specifically:
    1. Normalize team names across all relevant columns (so "Delhi
       Daredevils" and "Delhi Capitals" are treated as the same team).
    2. Handle the 26 matches with no `winner` (ties / no-results) -- kept
       for analytics, excluded from anything used as an ML training target.
    3. Derive simple date-based columns (year, month, day_of_week) from the
       `date` column for trend analysis.
    4. Run sanity-check validations (duplicate match IDs, toss_winner
       being one of the two competing teams, etc.) and log anything
       suspicious rather than silently proceeding.
    5. Persist cleaned DataFrames to data/processed/ so downstream steps
       (feature engineering, the Streamlit app) never touch raw data
       directly.

Usage:
    from src.data.preprocess import preprocess_all

    cleaned = preprocess_all()
    matches_clean = cleaned["matches"]

Or from the command line:
    python -m src.data.preprocess
"""

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

# Columns across the four datasets that contain team names and therefore
# need TEAM_NAME_MAPPING applied. Centralized here so adding a new dataset
# later only requires extending this list, not hunting through the file.
MATCHES_TEAM_COLUMNS: list[str] = ["team1", "team2", "toss_winner", "winner"]
DELIVERIES_TEAM_COLUMNS: list[str] = ["batting_team", "bowling_team"]
SEASONS_TEAM_COLUMNS: list[str] = ["champion", "runner_up"]


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


def preprocess_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for matches.csv.

    Steps applied in order:
        1. Normalize team names (team1, team2, toss_winner, winner).
        2. Flag no-result/tied matches explicitly.
        3. Add derived date features (year, month, day of week).
        4. Validate the cleaned data and log any issues found.

    Args:
        matches_df: Raw matches DataFrame from load_matches().

    Returns:
        Cleaned matches DataFrame, ready for feature engineering.
    """
    logger.info("Preprocessing matches.csv...")
    df = normalize_team_names(matches_df, MATCHES_TEAM_COLUMNS)
    df = flag_no_result_matches(df)
    df = add_date_features(df)
    validate_matches(df)
    logger.info(f"matches.csv preprocessing complete: {df.shape[0]:,} rows retained.")
    return df


def preprocess_deliveries(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for deliveries.csv.

    Steps applied:
        1. Normalize team names (batting_team, bowling_team).

    Deliveries data is otherwise already clean and ball-level granular --
    no date features or no-result flags apply at this grain (that
    information lives on the parent match, joinable via match_id).

    Args:
        deliveries_df: Raw deliveries DataFrame from load_deliveries().

    Returns:
        Cleaned deliveries DataFrame, ready for feature engineering.
    """
    logger.info("Preprocessing deliveries.csv...")
    df = normalize_team_names(deliveries_df, DELIVERIES_TEAM_COLUMNS)
    logger.info(f"deliveries.csv preprocessing complete: {df.shape[0]:,} rows retained.")
    return df


def preprocess_players(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning pipeline for players.csv.

    players.csv has no team-name columns to normalize and minimal cleaning
    needs based on current schema inspection -- this function exists
    primarily so the pipeline has a consistent per-dataset entry point,
    and as a place to add player-specific cleaning logic later (e.g.
    standardizing nationality strings) without restructuring the pipeline.

    Args:
        players_df: Raw players DataFrame from load_players().

    Returns:
        Cleaned players DataFrame.
    """
    logger.info("Preprocessing players.csv...")
    df = players_df.copy()
    logger.info(f"players.csv preprocessing complete: {df.shape[0]:,} rows retained.")
    return df


def preprocess_seasons(seasons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning pipeline for seasons.csv.

    Normalizes team names in the champion/runner_up columns so season
    summaries are consistent with the team names used elsewhere in the
    pipeline (e.g. a season where "Deccan Chargers" won should show as
    "Sunrisers Hyderabad" to match matches.csv after normalization).

    Args:
        seasons_df: Raw seasons DataFrame from load_seasons().

    Returns:
        Cleaned seasons DataFrame.
    """
    logger.info("Preprocessing seasons.csv...")
    df = normalize_team_names(seasons_df, SEASONS_TEAM_COLUMNS)
    logger.info(f"seasons.csv preprocessing complete: {df.shape[0]:,} rows retained.")
    return df


def preprocess_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run the full preprocessing pipeline for all four datasets.

    Args:
        save: If True (default), write each cleaned DataFrame to its
            configured path in data/processed/. Set to False if you only
            want the DataFrames in memory (e.g. for testing).

    Returns:
        Dictionary with keys "matches", "deliveries", "players", "seasons",
        each mapping to its cleaned DataFrame.
    """
    ensure_directories_exist()
    raw = load_all()

    cleaned = {
        "matches": preprocess_matches(raw["matches"]),
        "deliveries": preprocess_deliveries(raw["deliveries"]),
        "players": preprocess_players(raw["players"]),
        "seasons": preprocess_seasons(raw["seasons"]),
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
