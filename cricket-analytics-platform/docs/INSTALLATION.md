# Installation Guide

This guide walks through setting up the **Cricket Analytics & Match Prediction Platform** from a fresh clone, including how to obtain the IPL dataset (which is intentionally **not** committed to this repository — see `.gitignore`).

---

## 1. Prerequisites

- Python 3.12 (check with `python --version` or `python3 --version`)
- Git
- A free [Kaggle](https://www.kaggle.com/) account (for dataset download)

---

## 2. Clone the repository

```bash
git clone https://github.com/<your-username>/cricket-analytics-platform.git
cd cricket-analytics-platform
```

---

## 3. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Download the IPL dataset from Kaggle

This project expects **two** CSV files in `data/raw/` — note that this is fewer than earlier versions of this project expected, since this dataset does not ship separate player-metadata or season-summary files (see the note below the table):

| File | Description |
|------|-------------|
| `matches.csv` | One row per match: `matchId`, `season`, `venue`, `city`, `team1`, `team2`, `toss_winner`, `toss_decision`, `winner`, `winner_runs`, `winner_wickets`, `outcome`, `player_of_match`, `umpire1`, `umpire2`, etc. |
| `deliveries.csv` | One row per ball bowled: `matchId`, `inning`, `over`, `ball`, `batting_team`, `bowling_team`, `batsman`, `non_striker`, `bowler`, `batsman_runs`, `extras`, `isWide`/`isNoBall`/`Byes`/`LegByes`/`Penalty`, `dismissal_kind`, `player_dismissed`. |

These come from the **IPL ball-by-ball dataset (updated through 2025)** on Kaggle. Other "IPL" dataset uploads may also work as long as they provide match-level and ball-by-ball CSVs, since `src/data/loader.py` centralizes the expected raw column names (`EXPECTED_MATCHES_COLUMNS` / `EXPECTED_DELIVERIES_COLUMNS`) and `src/data/preprocess.py` centralizes the raw-to-canonical translation — but you will need to update both if a different version uses different headers.

> **No `players.csv` or `seasons.csv` needed.** Unlike an earlier version of this project, this dataset does not ship standalone player-metadata or season-summary files. Player rosters and season summaries are automatically **derived** from `matches.csv` + `deliveries.csv` during preprocessing (see `src/data/preprocess.py::derive_players()` / `derive_seasons()`) and written to `data/processed/players_clean.csv` / `seasons_clean.csv`. This means fields like player nationality, batting/bowling style, playing role, and auction price are **not available**, since that information doesn't exist anywhere in the source data — see the Dataset section of the main [`README.md`](../README.md) for details.

### Option A — Manual download (simplest)

1. Go to Kaggle and search for an **IPL ball-by-ball dataset covering 2008–2025** (e.g. "IPL Complete Dataset up to 2025").
2. Download the dataset `.zip` file.
3. Extract it and place `matches.csv` and `deliveries.csv` directly into `data/raw/`. (If the download uses different filenames, such as `matches_updated_ipl_upto_2025.csv`, rename them to `matches.csv` and `deliveries.csv` respectively.)

Your folder should look like:

```
data/raw/
├── matches.csv
├── deliveries.csv
└── .gitkeep
```

### Option B — Kaggle CLI (recommended for reproducibility)

1. Install the Kaggle CLI (not included in `requirements.txt` by default — install separately if you want this route):

   ```bash
   pip install kaggle
   ```

2. Get your API token:
   - Go to your Kaggle account settings → **API** → **Create New Token**.
   - This downloads a `kaggle.json` file containing your credentials.

3. Place the token where the CLI expects it:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   > ⚠️ **Never commit `kaggle.json` to Git.** It contains a secret API key. This repo's `.gitignore` does not reference it directly, so be careful to keep it outside the project folder or add `kaggle.json` to your global gitignore.

4. Download the dataset directly into `data/raw/`, then rename the files to match what this pipeline expects:

   ```bash
   kaggle datasets download -d <dataset-owner>/<dataset-slug> -p data/raw --unzip
   mv data/raw/matches_updated_ipl_upto_2025.csv data/raw/matches.csv
   mv data/raw/deliveries_updated_ipl_upto_2025.csv data/raw/deliveries.csv
   ```

   (Adjust the source filenames above to whatever your specific Kaggle download actually names them.)

5. Verify the files landed correctly:

   ```bash
   ls data/raw/
   # Expect: matches.csv  deliveries.csv  .gitkeep
   ```

### Option C — Cricsheet.org (alternative source)

[Cricsheet](https://cricsheet.org/) provides free ball-by-ball cricket data in YAML/JSON/CSV formats, including IPL matches, as an alternative or supplement to Kaggle. If you use Cricsheet data instead, you will need to adapt `EXPECTED_MATCHES_COLUMNS` / `EXPECTED_DELIVERIES_COLUMNS` in `src/data/loader.py` and the corresponding renaming logic in `src/data/preprocess.py` to match Cricsheet's schema, since it differs from the Kaggle format this project is built around.

---

## 6. Run the pipeline

Once `matches.csv` and `deliveries.csv` are in `data/raw/`:

```bash
# Clean and preprocess raw data (also derives players_clean.csv and seasons_clean.csv)
python -m src.data.preprocess

# Train and compare pre-match win-prediction ML models
python -m src.models.train

# Train and compare in-match ("live") win-probability models
python -m src.models.live_win_probability

# Launch the Streamlit dashboard
streamlit run streamlit_app/Home.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---------|----------------------|
| `FileNotFoundError: data/raw/matches.csv` | Dataset not downloaded yet, or files weren't renamed to `matches.csv`/`deliveries.csv` — see Step 5 above. |
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the project root, not from inside `src/`, so Python can resolve the `src` package. |
| Kaggle CLI `401 Unauthorized` | Your `kaggle.json` token is missing, expired, or in the wrong location (`~/.kaggle/kaggle.json`). |
| `DataValidationError: matches.csv is missing expected columns` | Your dataset version uses different column names than expected — check `EXPECTED_MATCHES_COLUMNS` / `EXPECTED_DELIVERIES_COLUMNS` in `src/data/loader.py` and update them (plus the renaming logic in `src/data/preprocess.py`) to match your actual CSV headers. |
| `ModuleNotFoundError: No module named 'xgboost'` (during `python -m src.models.train` or `python -m src.models.live_win_probability`) | Optional dependency — `pip install xgboost`. Training still works without it; XGBoost is simply skipped from the model comparison. |
| Live Win Probability page shows "no trained live model found" | Run `python -m src.models.live_win_probability` — it's a separate training step from `python -m src.models.train` (see Step 6 above), since it's a different model on a different (ball-level) feature set. |

---

Next: see [`docs/USER_GUIDE.md`](USER_GUIDE.md) for how to use the dashboard once it's running.
