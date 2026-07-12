"""
loader.py
=========
Raw data loading layer for the Cricket Analytics & Match Prediction Platform.

This module is responsible for ONE thing: reading the raw CSV files
(matches.csv, deliveries.csv) into pandas DataFrames, with validation,
type coercion, and logging -- but with NO cleaning, renaming, or feature
engineering logic. That separation matters: if loading and cleaning live
in the same function, debugging becomes much harder ("did this break
because the file is malformed, or because my cleaning logic has a bug?").
By isolating loading here, `src/data/preprocess.py` can assume it always
receives a DataFrame with the *raw* expected columns present, and is
itself responsible for translating that raw schema into the canonical
column names the rest of the pipeline expects (see `src/utils/config.py`).

------------------------------------------------------------------------
2026-07-01 DATASET MIGRATION NOTE
------------------------------------------------------------------------
This dataset version ships only TWO raw files -- matches.csv and
deliveries.csv -- with different headers than the previous dataset:

    matches.csv (raw headers):
        season, venue, event, winner_runs, umpire2, toss_winner, date,
        neutralvenue, umpire1, city, reserve_umpire, winner, eliminator,
        date1, method, team1, toss_decision, gender, team2,
        balls_per_over, winner_wickets, tv_umpire, player_of_match,
        match_referee, outcome, date2, match_number, matchId

    deliveries.csv (raw headers):
        matchId, inning, over_ball, over, ball, batting_team,
        bowling_team, batsman, non_striker, bowler, batsman_runs,
        extras, isWide, isNoBall, Byes, LegByes, Penalty,
        dismissal_kind, player_dismissed, date

There is no players.csv or seasons.csv in this dataset at all -- those
are now derived from matches + deliveries in preprocess.py instead of
loaded here. `load_players()` / `load_seasons()` and their EXPECTED_*
column sets have been removed accordingly; `load_all()` now returns only
"matches" and "deliveries".

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

from src.utils.config import RAW_DELIVERIES_PATH, RAW_MATCHES_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Expected RAW schemas -- used to fail loudly and early if the dataset
# version someone downloads doesn't match what this pipeline expects,
# rather than failing mysteriously several steps later. These describe
# the ACTUAL CSV headers on disk (verified against data/raw/matches.csv
# and data/raw/deliveries.csv on 2026-07-01), NOT the canonical/cleaned
# column names used elsewhere in the codebase -- the raw-to-canonical
# translation happens in `src/data/preprocess.py`.
# ------------------------------------------------------------------
EXPECTED_MATCHES_COLUMNS: set[str] = {
    "season", "venue", "event", "winner_runs", "umpire2", "toss_winner",
    "date", "neutralvenue", "umpire1", "city", "reserve_umpire", "winner",
    "eliminator", "date1", "method", "team1", "toss_decision", "gender",
    "team2", "balls_per_over", "winner_wickets", "tv_umpire",
    "player_of_match", "match_referee", "outcome", "date2",
    "match_number", "matchId",
}

EXPECTED_DELIVERIES_COLUMNS: set[str] = {
    "matchId", "inning", "over_ball", "over", "ball", "batting_team",
    "bowling_team", "batsman", "non_striker", "bowler", "batsman_runs",
    "extras", "isWide", "isNoBall", "Byes", "LegByes", "Penalty",
    "dismissal_kind", "player_dismissed", "date",
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
            fix `src/data/loader.py`'s EXPECTED_*_COLUMNS or their dataset
            without guessing.
    """
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise DataValidationError(
            f"{file_label} is missing expected columns: {sorted(missing)}. "
            f"This usually means your dataset version uses different "
            f"column names. Update EXPECTED_MATCHES_COLUMNS / "
            f"EXPECTED_DELIVERIES_COLUMNS in src/data/loader.py (and the "
            f"corresponding translation logic in src/data/preprocess.py) "
            f"to match your actual CSV headers."
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
    Load and validate the raw matches CSV (one row per IPL match).

    Args:
        path: Path to matches.csv. Defaults to the configured raw data path.

    Returns:
        DataFrame with RAW match-level columns straight off disk (season,
        venue, event, winner_runs, toss_winner, date, city, winner,
        eliminator, method, team1, toss_decision, gender, team2,
        balls_per_over, winner_wickets, player_of_match, umpire1,
        umpire2, matchId, etc. -- see `EXPECTED_MATCHES_COLUMNS`).
        The `date` column is parsed into a proper datetime dtype here (the
        one type coercion this loader performs, since pandas reads dates as
        plain strings by default and almost every downstream feature -- e.g.
        season trends, day-of-week effects -- depends on it being a real
        datetime). Renaming raw columns to the canonical schema (e.g.
        `matchId` -> `match_id`) happens in `src/data/preprocess.py`, not
        here, to keep loading and cleaning concerns separate.

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
            f"(ties or no-result matches, per the 'outcome' column) -- "
            f"these are NOT dropped here, but must be handled explicitly "
            f"in preprocessing before model training, since the target "
            f"column can't contain NaN."
        )

    return df


def load_deliveries(path: Path = RAW_DELIVERIES_PATH) -> pd.DataFrame:
    """
    Load and validate the raw deliveries CSV (one row per ball bowled).

    Args:
        path: Path to deliveries.csv. Defaults to the configured raw path.

    Returns:
        DataFrame with RAW ball-by-ball columns straight off disk: matchId,
        inning, over, ball, batting_team, bowling_team, batsman,
        non_striker, bowler, batsman_runs, extras, the per-extra-type flag
        columns (isWide/isNoBall/Byes/LegByes/Penalty), dismissal_kind,
        player_dismissed, date -- see `EXPECTED_DELIVERIES_COLUMNS`.
        Renaming/deriving canonical columns (e.g. `batsman` -> `striker`,
        deriving `total_runs` and `extra_type`) happens in
        `src/data/preprocess.py`, not here.

    Raises:
        FileNotFoundError: If deliveries.csv doesn't exist at `path`.
        DataValidationError: If expected columns are missing.
    """
    df = _read_csv(path, "deliveries.csv")
    _validate_columns(df, EXPECTED_DELIVERIES_COLUMNS, "deliveries.csv")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """
    Load both raw datasets in one call.

    Returns:
        Dictionary with keys "matches" and "deliveries", each mapping to
        its respective raw DataFrame.

        NOTE: unlike the previous dataset version, this no longer includes
        "players" or "seasons" keys -- this dataset ships no raw
        players.csv/seasons.csv to load. Those DataFrames are now built by
        `src/data/preprocess.py::derive_players()` and `derive_seasons()`
        from the cleaned matches + deliveries data instead, and are added
        to the dict returned by `preprocess_all()` (not this function).

    Example:
        >>> data = load_all()
        >>> data["matches"].shape
        (1169, 27)
    """
    logger.info("Loading raw datasets...")
    data = {
        "matches": load_matches(),
        "deliveries": load_deliveries(),
    }
    logger.info("All raw datasets loaded successfully.")
    return data


if __name__ == "__main__":
    # Allows running `python src/data/loader.py` directly as a quick sanity
    # check that both raw files load and validate correctly.
    datasets = load_all()
    for name, frame in datasets.items():
        print(f"{name}: {frame.shape}")
