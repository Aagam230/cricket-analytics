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

This project expects four CSV files in `data/raw/`:

| File | Description |
|------|-------------|
| `matches.csv` | One row per match: teams, venue, toss, winner, season, etc. |
| `deliveries.csv` | One row per ball bowled: striker, non-striker, bowler, runs, wickets, etc. |
| `players.csv` | One row per player: nationality, role, batting/bowling style, auction prices, career span. |
| `seasons.csv` | One row per IPL season: champion, runner-up, orange/purple cap winners, aggregate stats. |

These come from the **IPL Complete Dataset (2008–2025, Enhanced Edition)** on Kaggle (`meruvakodandasuraj/ipl-complete-dataset-2008-2025-enhanced-edition`), which ships all four files with column names matching this pipeline's expectations out of the box. Other "IPL Complete Dataset" uploads may also work if they include all four files, since `src/utils/config.py` centralizes column-name mapping — but you may need to update the column constants there if a different version uses different headers.

### Option A — Manual download (simplest)

1. Go to Kaggle and search **"IPL Complete Dataset 2008-2025 Enhanced Edition"** (or use the link above).
2. Download the dataset `.zip` file.
3. Extract it and place `matches.csv`, `deliveries.csv`, `players.csv`, and `seasons.csv` directly into `data/raw/`.

Your folder should look like:

```
data/raw/
├── matches.csv
├── deliveries.csv
├── players.csv
├── seasons.csv
└── .gitkeep
```

### Option B — Kaggle CLI (recommended for reproducibility)

1. Install the Kaggle CLI (already in `requirements.txt` is *not* included by default — install separately if you want this route):

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

4. Download and unzip the dataset directly into `data/raw/`:

   ```bash
   kaggle datasets download -d meruvakodandasuraj/ipl-complete-dataset-2008-2025-enhanced-edition -p data/raw --unzip
   ```

5. Verify the files landed correctly:

   ```bash
   ls data/raw/
   # Expect: matches.csv  deliveries.csv  players.csv  seasons.csv  .gitkeep
   ```

### Option C — Cricsheet.org (alternative source)

[Cricsheet](https://cricsheet.org/) provides free ball-by-ball cricket data in YAML/JSON/CSV formats, including IPL matches, as an alternative or supplement to Kaggle. If you use Cricsheet data instead, you may need to adapt column names in `src/utils/config.py` to match Cricsheet's schema, since it differs slightly from the Kaggle format this project is built around.

---

## 6. Run the pipeline

Once `matches.csv`, `deliveries.csv`, `players.csv`, and `seasons.csv` are in `data/raw/`:

```bash
# Clean and preprocess raw data
python -m src.data.preprocess

# Train and compare ML models
python -m src.models.train

# Launch the Streamlit dashboard
streamlit run streamlit_app/Home.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---------|----------------------|
| `FileNotFoundError: data/raw/matches.csv` | Dataset not downloaded yet — see Step 5 above. |
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the project root, not from inside `src/`, so Python can resolve the `src` package. |
| Kaggle CLI `401 Unauthorized` | Your `kaggle.json` token is missing, expired, or in the wrong location (`~/.kaggle/kaggle.json`). |
| Column name errors during preprocessing | Your Kaggle dataset version uses different column names than expected — check `src/utils/config.py` and update the column constants to match your CSV headers. |

---

Next: see [`docs/USER_GUIDE.md`](USER_GUIDE.md) for how to use the dashboard once it's running.
