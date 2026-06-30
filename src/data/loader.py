"""
loader.py
=========
Raw data loading layer for the Cricket Analytics & Match Prediction Platform.

This module is responsible for ONE thing: reading the raw Kaggle CSV files
(matches.csv, deliveries.csv, players.csv, seasons.csv) into pandas
DataFrames, with validation, type coercion, and logging -- but with NO
cleaning or feature engineering logic. That separation matters: if loading
and cleaning live in the same function, debugging becomes much harder
("did this break because the file is malformed, or because my cleaning
logic has a bug?"). By isolating loading here, `src/data/preprocess.py`
can assume it always receives a DataFrame with the expected columns
present and basic types coerced correctly.

Usage:
    from src.data.loader import load_matches, load_deliveries, load_all

    matches_df = load_matches()
    deliveries_df = load_deliveries()

    # or load everything at once:
    data = load_all()
    matches_df = data["matches"]
"""

from pathlib import Path

import pandas as pd

from src.utils.config import (
    RAW_DELIVERIES_PATH,
    RAW_MATCHES_PATH,
    RAW_PLAYERS_PATH,
    RAW_SEASONS_PATH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Expected schemas -- used to fail loudly and early if the dataset
# version someone downloads doesn't match what this pipeline expects,
# rather than failing mysteriously several steps later.
# ------------------------------------------------------------------
EXPECTED_MATCHES_COLUMNS: set[str] = {
    "match_id", "season", "match_number", "stage", "date", "venue", "city",
    "team1", "team2", "toss_winner", "toss_decision", "first_innings_score",
    "first_innings_wickets", "first_innings_overs", "second_innings_score",
    "second_innings_wickets", "second_innings_overs", "result", "winner",
    "win_by", "win_margin", "player_of_match", "umpire1", "umpire2",
    "is_day_night",
}

EXPECTED_DELIVERIES_COLUMNS: set[str] = {
    "delivery_id", "match_id", "innings", "over", "ball", "batting_team",
    "bowling_team", "striker", "non_striker", "bowler", "batsman_runs",
    "extra_runs", "total_runs", "extra_type", "is_wicket", "dismissal_type",
    "dismissed_player", "fielder",
}

EXPECTED_PLAYERS_COLUMNS: set[str] = {
    "player_id", "player_name", "nationality", "dob_year", "batting_style",
    "bowling_style", "playing_role", "ipl_debut_season",
    "last_season_played", "is_capped_international", "base_price_lakh",
    "highest_auction_price_lakh",
}

EXPECTED_SEASONS_COLUMNS: set[str] = {
    "season", "total_matches", "num_teams", "champion", "runner_up",
    "orange_cap_winner", "purple_cap_winner", "most_valuable_player",
    "total_runs_scored", "total_sixes", "total_fours",
    "avg_first_innings_score", "highest_team_total", "lowest_team_total",
}


class DataValidationError(Exception):
    """
    Raised when a loaded CSV's columns don't match the expected schema.

    Using a custom exception (rather than a generic ValueError) makes it
    possible for calling code to catch THIS specific failure mode
    distinctly from other errors, and makes error messages in logs/
    tracebacks immediately identifiable as a schema problem rather than
    some other kind of bug.
    """
    pass


def _validate_columns(df: pd.DataFrame, expected: set[str], file_label: str) -> None:
    """
    Verify that a loaded DataFrame contains all expected columns.

    Args:
        df: The loaded DataFrame to validate.
        expected: Set of column names that must be present.
        file_label: Human-readable name of the file, used in error messages.

    Raises:
        DataValidationError: If any expected column is missing. The error
            message lists exactly which columns are missing so the user can
            fix `src/utils/config.py` or their dataset without guessing.
    """
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise DataValidationError(
            f"{file_label} is missing expected columns: {sorted(missing)}. "
            f"This usually means your Kaggle dataset version uses different "
            f"column names. Update the corresponding constants in "
            f"src/utils/config.py to match your actual CSV headers."
        )

    extra = actual - expected
    if extra:
        # Not a fatal error -- extra columns are fine, just worth knowing
        # about in case the dataset has bonus fields we could use later.
        logger.info(f"{file_label} has extra columns not used by the pipeline: {sorted(extra)}")


def _read_csv(path: Path, file_label: str) -> pd.DataFrame:
    """
    Read a CSV file with consistent error handling and logging.

    Args:
        path: Path to the CSV file.
        file_label: Human-readable name used in log messages and errors.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist at `path`, with a
            message pointing the user to docs/INSTALLATION.md.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{file_label} not found at {path}. "
            f"See docs/INSTALLATION.md for instructions on downloading the "
            f"IPL dataset and placing it in data/raw/."
        )

    logger.info(f"Loading {file_label} from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {file_label}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_matches(path: Path = RAW_MATCHES_PATH) -> pd.DataFrame:
    """
    Load and validate matches.csv (one row per IPL match).

    Args:
        path: Path to matches.csv. Defaults to the configured raw data path.

    Returns:
        DataFrame with match-level data: teams, venue, toss, winner, etc.
        The `date` column is parsed into a proper datetime dtype here (the
        one type coercion this loader performs, since pandas reads dates as
        plain strings by default and almost every downstream feature -- e.g.
        season trends, day-of-week effects -- depends on it being a real
        datetime).

    Raises:
        FileNotFoundError: If matches.csv doesn't exist at `path`.
        DataValidationError: If expected columns are missing.
    """
    df = _read_csv(path, "matches.csv")
    _validate_columns(df, EXPECTED_MATCHES_COLUMNS, "matches.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    n_missing_winner = df["winner"].isna().sum()
    if n_missing_winner > 0:
        logger.warning(
            f"matches.csv has {n_missing_winner} rows with no winner "
            f"(likely ties or no-result matches) -- these are NOT dropped "
            f"here, but must be handled explicitly in preprocessing before "
            f"model training, since the target column can't contain NaN."
        )

    return df


def load_deliveries(path: Path = RAW_DELIVERIES_PATH) -> pd.DataFrame:
    """
    Load and validate deliveries.csv (one row per ball bowled).

    Args:
        path: Path to deliveries.csv. Defaults to the configured raw path.

    Returns:
        DataFrame with ball-by-ball data: batting/bowling team, striker,
        bowler, runs, wickets, etc.

    Raises:
        FileNotFoundError: If deliveries.csv doesn't exist at `path`.
        DataValidationError: If expected columns are missing.
    """
    df = _read_csv(path, "deliveries.csv")
    _validate_columns(df, EXPECTED_DELIVERIES_COLUMNS, "deliveries.csv")
    return df


def load_players(path: Path = RAW_PLAYERS_PATH) -> pd.DataFrame:
    """
    Load and validate players.csv (one row per player).

    Args:
        path: Path to players.csv. Defaults to the configured raw path.

    Returns:
        DataFrame with player metadata: nationality, role, batting/bowling
        style, auction prices, career span.

    Raises:
        FileNotFoundError: If players.csv doesn't exist at `path`.
        DataValidationError: If expected columns are missing.
    """
    df = _read_csv(path, "players.csv")
    _validate_columns(df, EXPECTED_PLAYERS_COLUMNS, "players.csv")
    return df


def load_seasons(path: Path = RAW_SEASONS_PATH) -> pd.DataFrame:
    """
    Load and validate seasons.csv (one row per IPL season).

    Args:
        path: Path to seasons.csv. Defaults to the configured raw path.

    Returns:
        DataFrame with season-level summaries: champion, runner-up, cap
        winners, aggregate run/six/four counts.

    Raises:
        FileNotFoundError: If seasons.csv doesn't exist at `path`.
        DataValidationError: If expected columns are missing.
    """
    df = _read_csv(path, "seasons.csv")
    _validate_columns(df, EXPECTED_SEASONS_COLUMNS, "seasons.csv")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """
    Load all four raw datasets in one call.

    Returns:
        Dictionary with keys "matches", "deliveries", "players", "seasons",
        each mapping to its respective DataFrame.

    Example:
        >>> data = load_all()
        >>> data["matches"].shape
        (1158, 25)
    """
    logger.info("Loading all raw datasets...")
    data = {
        "matches": load_matches(),
        "deliveries": load_deliveries(),
        "players": load_players(),
        "seasons": load_seasons(),
    }
    logger.info("All raw datasets loaded successfully.")
    return data


if __name__ == "__main__":
    # Allows running `python src/data/loader.py` directly as a quick sanity
    # check that all four files load and validate correctly.
    datasets = load_all()
    for name, frame in datasets.items():
        print(f"{name}: {frame.shape}")
