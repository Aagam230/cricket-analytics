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
RAW_PLAYERS_PATH: Path = RAW_DATA_DIR / "players.csv"
RAW_SEASONS_PATH: Path = RAW_DATA_DIR / "seasons.csv"

PROCESSED_MATCHES_PATH: Path = PROCESSED_DATA_DIR / "matches_clean.csv"
PROCESSED_DELIVERIES_PATH: Path = PROCESSED_DATA_DIR / "deliveries_clean.csv"
PROCESSED_PLAYERS_PATH: Path = PROCESSED_DATA_DIR / "players_clean.csv"
PROCESSED_SEASONS_PATH: Path = PROCESSED_DATA_DIR / "seasons_clean.csv"
MODEL_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "model_features.csv"

# ------------------------------------------------------------------
# Model artifact paths
# ------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
ENCODERS_PATH: Path = MODELS_DIR / "encoders.joblib"

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
# Target / key column names — matches.csv
# ------------------------------------------------------------------
# These match the ACTUAL schema of the uploaded Kaggle dataset
# (verified against data/raw/matches.csv on 2026-06-30). Centralized here
# so that if a different dataset version uses different names, only this
# file needs to change.
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
IS_DAY_NIGHT_COLUMN: str = "is_day_night"

# ------------------------------------------------------------------
# deliveries.csv columns
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
FIELDER_COLUMN: str = "fielder"

# ------------------------------------------------------------------
# players.csv columns
# ------------------------------------------------------------------
PLAYER_ID_COLUMN: str = "player_id"
PLAYER_NAME_COLUMN: str = "player_name"
NATIONALITY_COLUMN: str = "nationality"
DOB_YEAR_COLUMN: str = "dob_year"
BATTING_STYLE_COLUMN: str = "batting_style"
BOWLING_STYLE_COLUMN: str = "bowling_style"
PLAYING_ROLE_COLUMN: str = "playing_role"
IPL_DEBUT_SEASON_COLUMN: str = "ipl_debut_season"
LAST_SEASON_PLAYED_COLUMN: str = "last_season_played"
IS_CAPPED_INTERNATIONAL_COLUMN: str = "is_capped_international"
BASE_PRICE_LAKH_COLUMN: str = "base_price_lakh"
HIGHEST_AUCTION_PRICE_LAKH_COLUMN: str = "highest_auction_price_lakh"

# ------------------------------------------------------------------
# seasons.csv columns
# ------------------------------------------------------------------
TOTAL_MATCHES_COLUMN: str = "total_matches"
NUM_TEAMS_COLUMN: str = "num_teams"
CHAMPION_COLUMN: str = "champion"
RUNNER_UP_COLUMN: str = "runner_up"
ORANGE_CAP_WINNER_COLUMN: str = "orange_cap_winner"
PURPLE_CAP_WINNER_COLUMN: str = "purple_cap_winner"

# ------------------------------------------------------------------
# Team name normalization
# ------------------------------------------------------------------
# IPL team names/franchises have changed over the years (e.g. relocations,
# rebrands, sponsor changes). This mapping standardizes historical names to
# their current equivalents so the model doesn't treat "Delhi Daredevils"
# and "Delhi Capitals" as two unrelated teams.
#
# NOTE: "Pune Warriors India" (2011-2013) and "Rising Pune Supergiant"
# (2016-2017) are DELIBERATELY NOT merged here -- despite both being
# Pune-based, they were distinct franchises under different ownership/
# names, not a rename of the same team. Merging them would incorrectly
# conflate two different entities' historical performance.
#
# Verified against actual unique team names in data/raw/matches.csv
# (17 unique team strings, 2026-06-30).
TEAM_NAME_MAPPING: dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
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
