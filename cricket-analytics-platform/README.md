# 🏏 Cricket Analytics & Match Prediction Platform

> An end-to-end machine learning platform for IPL match outcome prediction and cricket analytics, built with a production-style Python pipeline and an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project predicts the outcome of IPL (Indian Premier League) cricket matches using historical match and ball-by-ball data, and surfaces rich team/player/venue analytics through an interactive dashboard.

It was built to demonstrate an end-to-end, production-style data science workflow: data ingestion → cleaning → feature engineering → model training & comparison → deployment, rather than a single notebook.

**Status:** 🚧 In active development (built incrementally, module by module).

---

## ✨ Features

- **Match Outcome Prediction** — predicts the winner of a hypothetical IPL match with a win probability and confidence score, based on teams, venue, toss, and season.
- **Player Performance Dashboard** — top run scorers, wicket takers, batting/bowling averages, strike rates, economy rates, season-wise leaders.
- **Team Analytics** — win percentages, venue performance, toss impact, head-to-head records, home vs. away splits.
- **Match Insights** — chasing vs. defending trends, venue statistics, seasonal trends.
- **Model Comparison** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost evaluated side by side; best model auto-selected and persisted.
- **Interactive Streamlit App** — multi-page dashboard (Home, Match Prediction, Team Analytics, Player Analytics, Visualizations, Model Performance, About).

---

## 📷 Screenshots

> _To be added as the Streamlit app is built._

| Home | Match Prediction | Analytics |
|------|-------------------|-----------|
| _placeholder_ | _placeholder_ | _placeholder_ |

---

## 🗂️ Project Architecture

```
cricket-analytics-platform/
├── data/
│   ├── raw/              # Original Kaggle/Cricsheet CSVs
│   └── processed/        # Cleaned, feature-engineered datasets
├── models/                # Trained model artifacts (.joblib) + metadata
├── src/
│   ├── data/              # Data loading & cleaning
│   ├── features/          # Feature engineering
│   ├── models/             # Training, evaluation, model selection
│   └── utils/               # Logging, config, shared helpers
├── streamlit_app/
│   └── pages/                # Multi-page Streamlit dashboard
├── notebooks/                  # Exploratory analysis
├── visualizations/              # Saved charts
├── docs/                         # Installation guide, user guide, architecture notes
├── tests/                         # Unit tests
├── requirements.txt
└── README.md
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a detailed folder-by-folder explanation.

---

## 📊 Dataset

This project uses a real-world IPL ball-by-ball dataset covering every season from 2007/08 through 2025, sourced from Kaggle ("IPL Complete Dataset, updated through 2025"):

- `data/raw/matches.csv` — one row per match (1,169 matches), including season, venue, teams, toss, result, and umpires.
- `data/raw/deliveries.csv` — one row per ball bowled (~278,000 deliveries), including batter/bowler, runs, extras breakdown (wide/no-ball/bye/leg-bye/penalty), and dismissals.

Unlike an earlier version of this project, **this dataset does not ship separate `players.csv` or `seasons.csv` files** — player rosters and season summaries are derived automatically from the match and ball-by-ball data during preprocessing (`src/data/preprocess.py::derive_players()` / `derive_seasons()`). This means player metadata such as nationality, batting/bowling style, playing role, and auction price is **not available** in this version, since it doesn't exist anywhere in the source data.

Raw data is **not committed to this repository** (see `.gitignore`); instructions for downloading it are in [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

---

## 🤖 Model Explanation

Five classification models are trained and compared on engineered match features:

| Model | Why it's included |
|-------|---------------------|
| Logistic Regression | Simple, interpretable baseline |
| Decision Tree | Captures non-linear splits, easy to visualize |
| Random Forest | Reduces overfitting via bagging, handles feature interactions |
| Gradient Boosting | Sequential error correction, strong baseline performance |
| XGBoost | Industry-standard gradient boosting, typically best performance |

Models are evaluated with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and k-fold cross-validation. The best-performing model (by cross-validated F1) is automatically selected and persisted to `models/`.

**Evaluation methodology:** the test set is a **time-based split** — the most recent ~20% of matches, chronologically — rather than a random sample, since a model meant to predict *future* matches shouldn't be scored on its ability to predict matches from a decade before its "test" data. This is more honest but also harder: cross-validated F1 sits around 0.51–0.53 and test-set ROC-AUC hovers close to 0.5 for most models. That's a genuine finding, not a bug — pre-match features (teams, venue, toss, historical form) capture only part of what decides an IPL match; a large share of the outcome is determined by what happens *during* the game (which this feature set intentionally excludes to avoid leakage).

Full methodology, per-model metrics tables, and an honest discussion of why performance sits where it does: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) (also summarized in the in-app **About** page).

### In-Match ("Live") Win Probability — a follow-up model

Given the pre-match model's honest, close-to-chance performance above, a second model was built to test whether that reflects a modeling limitation or an information limitation: predicting win probability **during** a second-innings chase from live match state (target, current score, wickets, overs remaining) rather than pre-match information. Evaluated with the same rigor (match-grouped cross-validation, chronological match-level test split — see `docs/MODEL_CARD.md` Part 2 for why this matters), it reaches **test ROC-AUC 0.876** — a large, genuine improvement over the pre-match model's 0.538, confirming that most of what decides an IPL match isn't knowable before it starts, but the state of the game itself is a strong predictor once play is underway. Try it on the **Live Win Probability** page — replay a real historical chase ball-by-ball, or enter a hypothetical match state for an instant estimate. Full methodology in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), Part 2.

---

## ⚙️ Installation

See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) for full setup instructions. Quick start:

```bash
git clone https://github.com/<your-username>/cricket-analytics-platform.git
cd cricket-analytics-platform
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

> **Note:** Always run pipeline scripts as modules from the project root (e.g. `python -m src.data.loader`), not directly (`python src/data/loader.py`), so Python can resolve the `src` package correctly.

```bash
# 1. Place raw CSVs in data/raw/ (see docs/INSTALLATION.md)
# 2. Run the data pipeline
python -m src.data.preprocess

# 3. Train and compare pre-match win-prediction models
python -m src.models.train

# 4. Train and compare in-match ("live") win-probability models
python -m src.models.live_win_probability

# 5. Launch the dashboard
streamlit run streamlit_app/Home.py
```

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for a full walkthrough.

---

## 🔮 Future Improvements

- Live data ingestion from current IPL season APIs
- Ball-by-ball in-match win probability prediction
- Player form/momentum-based features
- Model explainability (SHAP values) in the dashboard
- Deployment to Streamlit Community Cloud / Docker

---

## 🛠️ Tech Stack

Python 3.12 · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Plotly · Streamlit · Joblib

---

## 📄 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
