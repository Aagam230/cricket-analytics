"""
config.py
==========
Centralized configuration for the Cricket Analytics & Match Prediction Platform.

This module defines all project-wide constants in ONE place: file paths,
model hyperparameter defaults, column name mappings, and reproducibility
settings. Every other module (data loading, feature engineering, training,
the Streamlit app) imports from here instead of hardcoding paths or magic
numbers.

Why this matters:
    Hardcoding paths like "data/raw/matches.csv" all over the codebase is a
    common beginner mistake -- it makes the project fragile (one renamed
    folder breaks ten files) and hard to reproduce on another machine. By
    centralizing configuration, we only need to change a value in ONE place
    if, say, we move the project, rename a file, or change a model's random
    seed.

------------------------------------------------------------------------
2026-07-01 DATASET MIGRATION NOTE
------------------------------------------------------------------------
This project was migrated from a 4-file synthetic dataset
(matches.csv / deliveries.csv / players.csv / seasons.csv) to a 2-file
real-world IPL dataset (matches.csv / deliveries.csv only, sourced from
the "IPL ball-by-ball + match" CSVs covering 2008-2025). The new dataset
has NO standalone players.csv or seasons.csv -- those are now DERIVED
during preprocessing (see `src/data/preprocess.py::derive_players` and
`derive_seasons`) rather than loaded from raw files.

To keep every downstream module (feature engineering, analytics,
visualization) working UNCHANGED, `src/data/preprocess.py` translates the
new raw schema into the same CANONICAL column names this file has always
defined (e.g. `match_id`, `winner`, `first_innings_score`, `striker`,
`total_runs`, ...). The constants below therefore still describe the
*processed/cleaned* schema that the rest of the codebase consumes -- only
`src/data/loader.py`'s `EXPECTED_*_COLUMNS` (which describe the *raw* file
headers) changed to match the new dataset's actual headers.

One feature was dropped entirely because the new dataset has no
equivalent information anywhere: `is_day_night` (no day/night or match
start-time field exists in the new files, so this column is no longer
defined here and no longer appears in the engineered feature set).
"""

from pathlib import Path

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
# Path(__file__) -> .../cricket-analytics-platform/src/utils/config.py
# .parent.parent.parent climbs: utils -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RAW_MATCHES_PATH: Path = RAW_DATA_DIR / "matches.csv"
RAW_DELIVERIES_PATH: Path = RAW_DATA_DIR / "deliveries.csv"
# NOTE: there is no raw players.csv / seasons.csv anymore -- the new
# dataset doesn't ship them. `players_clean.csv` and `seasons_clean.csv`
# are now DERIVED from matches + deliveries in preprocess.py and written
# straight to PROCESSED_DATA_DIR (there is no "raw" counterpart to read).

PROCESSED_MATCHES_PATH: Path = PROCESSED_DATA_DIR / "matches_clean.csv"
PROCESSED_DELIVERIES_PATH: Path = PROCESSED_DATA_DIR / "deliveries_clean.csv"
PROCESSED_PLAYERS_PATH: Path = PROCESSED_DATA_DIR / "players_clean.csv"
PROCESSED_SEASONS_PATH: Path = PROCESSED_DATA_DIR / "seasons_clean.csv"
MODEL_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "model_features.csv"

# `live_state_features.csv` is the ball-by-ball (in-match) counterpart to
# `model_features.csv` -- see src/features/live_state_engineering.py and
# the "Part 2" section of docs/MODEL_CARD.md.
LIVE_MODEL_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "live_state_features.csv"

# ------------------------------------------------------------------
# Model artifact paths
# ------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
ENCODERS_PATH: Path = MODELS_DIR / "encoders.joblib"

# In-match ("live") win-probability model artifacts -- a SEPARATE model
# from the pre-match predictor above, trained on ball-by-ball chase state
# rather than pre-match team/venue/toss info. Kept as distinct files
# (rather than overwriting the pre-match model's artifacts) since both
# models coexist and are used for different purposes -- see
# src/models/live_win_probability.py and the "Part 2" section of
# docs/MODEL_CARD.md.
BEST_LIVE_MODEL_PATH: Path = MODELS_DIR / "best_live_model.joblib"
LIVE_MODEL_METADATA_PATH: Path = MODELS_DIR / "live_model_metadata.json"
LIVE_ENCODERS_PATH: Path = MODELS_DIR / "live_encoders.joblib"

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE_PATH: Path = LOGS_DIR / "pipeline.log"

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_SEED: int = 42

# ------------------------------------------------------------------
# Train/test split settings
# ------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ------------------------------------------------------------------
# Target / key column names -- CANONICAL (processed) matches schema
# ------------------------------------------------------------------
# These are the column names every module OTHER than loader.py works
# with -- i.e. the output of `src/data/preprocess.py`, not the raw CSV
# headers. preprocess.py is responsible for producing a DataFrame with
# exactly these columns regardless of what the raw dataset calls them.
TARGET_COLUMN: str = "winner"

MATCH_ID_COLUMN: str = "match_id"
SEASON_COLUMN: str = "season"
MATCH_NUMBER_COLUMN: str = "match_number"
STAGE_COLUMN: str = "stage"
DATE_COLUMN: str = "date"
VENUE_COLUMN: str = "venue"
CITY_COLUMN: str = "city"
TEAM1_COLUMN: str = "team1"
TEAM2_COLUMN: str = "team2"
TOSS_WINNER_COLUMN: str = "toss_winner"
TOSS_DECISION_COLUMN: str = "toss_decision"
FIRST_INNINGS_SCORE_COLUMN: str = "first_innings_score"
FIRST_INNINGS_WICKETS_COLUMN: str = "first_innings_wickets"
FIRST_INNINGS_OVERS_COLUMN: str = "first_innings_overs"
SECOND_INNINGS_SCORE_COLUMN: str = "second_innings_score"
SECOND_INNINGS_WICKETS_COLUMN: str = "second_innings_wickets"
SECOND_INNINGS_OVERS_COLUMN: str = "second_innings_overs"
RESULT_COLUMN: str = "result"
WIN_BY_COLUMN: str = "win_by"
WIN_MARGIN_COLUMN: str = "win_margin"
PLAYER_OF_MATCH_COLUMN: str = "player_of_match"
UMPIRE1_COLUMN: str = "umpire1"
UMPIRE2_COLUMN: str = "umpire2"
# IS_DAY_NIGHT_COLUMN was REMOVED here -- the new dataset has no
# day/night or match start-time field anywhere (raw matches.csv has no
# such column, and there is no way to derive it from ball-by-ball data
# either). Every module that used to consume "is_day_night" as a feature
# (src/features/engineering.py, src/models/predict.py) has been updated
# to no longer expect it. See the migration note at the top of this file.

# ------------------------------------------------------------------
# deliveries.csv columns -- CANONICAL (processed) schema
# ------------------------------------------------------------------
DELIVERY_ID_COLUMN: str = "delivery_id"
INNINGS_COLUMN: str = "innings"
OVER_COLUMN: str = "over"
BALL_COLUMN: str = "ball"
BATTING_TEAM_COLUMN: str = "batting_team"
BOWLING_TEAM_COLUMN: str = "bowling_team"
STRIKER_COLUMN: str = "striker"
NON_STRIKER_COLUMN: str = "non_striker"
BOWLER_COLUMN: str = "bowler"
BATSMAN_RUNS_COLUMN: str = "batsman_runs"
EXTRA_RUNS_COLUMN: str = "extra_runs"
TOTAL_RUNS_COLUMN: str = "total_runs"
EXTRA_TYPE_COLUMN: str = "extra_type"
IS_WICKET_COLUMN: str = "is_wicket"
DISMISSAL_TYPE_COLUMN: str = "dismissal_type"
DISMISSED_PLAYER_COLUMN: str = "dismissed_player"
# FIELDER_COLUMN: the new dataset has NO fielder-name field anywhere (not
# even in dismissal records) -- the closest available alternative,
# `dismissed_player` + `dismissal_type`, is preserved, but "who took the
# catch / effected the run out" is simply not recorded in this dataset.
# The constant is kept (for any external code that imports it) but the
# column it points to will always be NaN in `deliveries_clean.csv`.
FIELDER_COLUMN: str = "fielder"

# ------------------------------------------------------------------
# players_clean.csv columns -- DERIVED, not loaded from a raw file
# ------------------------------------------------------------------
# The new dataset ships no players.csv at all (no nationality, batting
# style, bowling style, playing role, auction price, or capped-status
# data exists anywhere in the source files). `derive_players()` in
# preprocess.py builds the closest meaningful alternative: a player
# roster reconstructed from ball-by-ball appearances, with debut/last
# season and basic participation counts -- everything else is
# unavailable and intentionally omitted rather than guessed.
PLAYER_ID_COLUMN: str = "player_id"
PLAYER_NAME_COLUMN: str = "player_name"
IPL_DEBUT_SEASON_COLUMN: str = "ipl_debut_season"
LAST_SEASON_PLAYED_COLUMN: str = "last_season_played"
MATCHES_BATTED_COLUMN: str = "matches_batted"
MATCHES_BOWLED_COLUMN: str = "matches_bowled"
# The following fields existed in the OLD dataset's players.csv but have
# NO equivalent data anywhere in the new dataset, so they are no longer
# defined: nationality, dob_year, batting_style, bowling_style,
# playing_role, is_capped_international, base_price_lakh,
# highest_auction_price_lakh.

# ------------------------------------------------------------------
# seasons_clean.csv columns -- DERIVED, not loaded from a raw file
# ------------------------------------------------------------------
# Like players.csv, the new dataset ships no seasons.csv. `derive_seasons()`
# in preprocess.py reconstructs an equivalent summary by aggregating the
# cleaned matches + deliveries data per season.
TOTAL_MATCHES_COLUMN: str = "total_matches"
NUM_TEAMS_COLUMN: str = "num_teams"
CHAMPION_COLUMN: str = "champion"
RUNNER_UP_COLUMN: str = "runner_up"
ORANGE_CAP_WINNER_COLUMN: str = "orange_cap_winner"
PURPLE_CAP_WINNER_COLUMN: str = "purple_cap_winner"
# `most_valuable_player` had no well-defined source in the old dataset
# either (it was a flat, un-derivable field); it is dropped here since
# there's no reliable way to compute "most valuable" from ball-by-ball
# data alone, and fabricating it would be misleading.

# ------------------------------------------------------------------
# Team name normalization
# ------------------------------------------------------------------
# IPL team names/franchises have changed over the years (e.g. relocations,
# rebrands, sponsor changes). This mapping standardizes historical names to
# their current equivalents so the model doesn't treat "Delhi Daredevils"
# and "Delhi Capitals" as two unrelated teams.
#
# NOTE: "Pune Warriors" (2011-2013) and "Rising Pune Supergiant(s)"
# (2016-2017) are DELIBERATELY NOT merged with each other -- despite both
# being Pune-based, they were distinct franchises under different
# ownership/names, not a rename of the same team. Merging them would
# incorrectly conflate two different entities' historical performance.
#
# Verified against actual unique team names in data/raw/matches.csv
# (18 unique team strings across both team1/team2, 2026-07-01).
TEAM_NAME_MAPPING: dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    # New in this dataset version (not present in the old synthetic data):
    # Royal Challengers Bangalore was officially rebranded to Royal
    # Challengers Bengaluru ahead of the 2024 season -- same franchise.
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    # The raw data spells this team inconsistently across seasons
    # ("Rising Pune Supergiants" in 2016, "Rising Pune Supergiant" --
    # singular -- in 2017). This is the SAME franchise both years, just a
    # data-entry inconsistency, unlike the Pune Warriors case above.
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

# ------------------------------------------------------------------
# Streamlit app settings
# ------------------------------------------------------------------
APP_TITLE: str = "Cricket Analytics & Match Prediction Platform"
APP_ICON: str = "🏏"
APP_LAYOUT: str = "wide"


def ensure_directories_exist() -> None:
    """
    Create all required project directories if they do not already exist.

    This is called once at the start of pipeline scripts (data preprocessing,
    training, etc.) so that the project works correctly even on a fresh
    clone of the repository where empty data/model/log folders aren't
    guaranteed to exist (since .gitignore excludes their contents).
    """
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
