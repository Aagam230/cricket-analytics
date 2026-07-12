"""
6_About.py
==========
Streamlit page: dataset provenance, methodology notes, and known
limitations -- so anyone using the dashboard understands what the numbers
do and don't mean, especially the gaps introduced by this dataset's
schema (no day/night flag, no player-metadata file, etc.).
"""

import _bootstrap  # noqa: F401

import streamlit as st

from data_service import get_matches
from src.utils.config import APP_ICON, APP_TITLE

st.set_page_config(page_title=f"About | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("ℹ️ About This Project")

st.markdown(
    """
## Overview

This project predicts the outcome of IPL (Indian Premier League) cricket
matches using historical match and ball-by-ball data, and surfaces team,
player, and venue analytics through this dashboard.

## Dataset

This dashboard is built on a two-file IPL dataset:

- **`matches.csv`** — one row per match (season, venue, teams, toss,
  result, umpires), covering every IPL season from 2007/08 through 2025.
- **`deliveries.csv`** — one row per ball bowled (batter, bowler, runs,
  extras breakdown, dismissals), roughly 278,000 rows.

Unlike some earlier IPL datasets, this one does **not** ship separate
`players.csv` or `seasons.csv` files. Both are automatically **derived**
from the match and ball-by-ball data during preprocessing:

- **Player roster** — reconstructed from every distinct name seen as a
  striker, non-striker, or bowler, with debut/last season and career
  totals computed from ball-by-ball appearances.
- **Season summaries** — aggregated per season from the cleaned match and
  delivery data. Champion/runner-up are *inferred* as the winner/loser of
  each season's chronologically last match (there's no explicit "this was
  the final" flag in the raw data), so treat those two fields as a
  well-informed inference rather than verified ground truth.

## Known Limitations

Because this dataset doesn't record certain information at all, a few
features present in earlier versions of this project are simply
unavailable here rather than approximated or guessed:

- **No day/night flag.** There's no match start-time or floodlit
  indicator anywhere in the source data, so the model has no
  `is_day_night` feature, and there's no day-vs-night scoring comparison
  on the Match Insights page.
- **No player metadata.** Nationality, batting/bowling style, playing
  role, capped/uncapped status, and auction prices don't exist anywhere
  in this dataset — the Player Analytics page only shows what's
  derivable from ball-by-ball data (career stats, debut/last season).
- **No fielder names.** Dismissal records include *who was dismissed*
  and *how*, but not *who took the catch* or *effected the run out*.
- **No tournament stage/phase field.** There's no explicit "league stage"
  vs. "playoffs" vs. "final" marker on individual matches.

## Model Methodology

Match outcome prediction uses only features that would be knowable
*before* the match is played (teams, venue, toss, season, and rolling
team-performance statistics computed strictly from *prior* matches — no
current-match information leaks into the feature set). Several
classification models are trained and compared via stratified k-fold
cross-validation plus a held-out **time-based test set** (the most recent
~20% of matches chronologically, not a random sample — see below); the
best model (by cross-validated F1 score) is automatically selected. See
the **Model Performance** page for the current comparison.

Two tuning passes have gone into the features so far: (1) win-rate
statistics are recency-weighted (exponential decay) rather than flat
career averages, since IPL squads turn over substantially every season
through trades and auctions — a flat "career win %" mixes in performance
from a roster that may no longer resemble the current team; and (2) venue
win percentages are shrunk toward a team's overall win rate based on
sample size, so two matches at a venue aren't treated with the same
confidence as forty.

**Honest performance note:** even after tuning, test-set ROC-AUC sits
close to 0.5–0.6 depending on the model, and cross-validated F1 hovers
around 0.51–0.53. This isn't a bug — it reflects a real property of the
prediction task: pre-match information (teams, venue, toss, historical
form) only explains part of what decides an IPL match. A meaningful share
of the outcome is determined by what happens *during* the game itself
(individual player form on the day, in-match momentum swings, weather,
etc.), which this feature set deliberately excludes to avoid leaking
future information into a pre-match prediction. A model that predicts
in-match win probability from live ball-by-ball state would likely
perform substantially better, since it has access to information that
simply doesn't exist before the first ball is bowled.

For the full methodology, per-model metrics tables, and confusion-matrix
level detail, see `docs/MODEL_CARD.md` in the project repository.

## Data Refresh

To refresh the dashboard with newer data, replace `data/raw/matches.csv`
and `data/raw/deliveries.csv`, then re-run:

```bash
python -m src.data.preprocess
python -m src.models.train
```
    """
)

try:
    matches_df = get_matches()
    st.caption(
        f"Currently loaded: {len(matches_df):,} matches spanning "
        f"{int(matches_df['season'].min())}–{int(matches_df['season'].max())}."
    )
except FileNotFoundError:
    pass
