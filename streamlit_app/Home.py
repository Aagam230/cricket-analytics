"""
Home.py
=======
Landing page for the Cricket Analytics & Match Prediction Platform's
Streamlit dashboard.

Run with:
    streamlit run streamlit_app/Home.py

This page gives a high-level overview of the dataset and links out to
each analysis page (Match Prediction, Team Analytics, Player Analytics,
Match Insights, Model Performance, About) via the sidebar, which
Streamlit's native multipage-app mechanism builds automatically from the
numbered files in `streamlit_app/pages/`.
"""

import _bootstrap  # noqa: F401  (side effect: fixes sys.path so `from src...` works)

import streamlit as st

from data_service import get_matches, get_players
from src.models.predict import ModelNotTrainedError, load_model_metadata
from src.utils.config import APP_ICON, APP_LAYOUT, APP_TITLE

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "An end-to-end machine learning platform for IPL match outcome "
    "prediction and cricket analytics."
)

st.divider()

try:
    matches_df = get_matches()
    players_df = get_players()
    data_loaded = True
except FileNotFoundError as exc:
    data_loaded = False
    st.error(
        f"Could not load the IPL dataset: {exc}\n\n"
        f"See `docs/INSTALLATION.md` for instructions on downloading "
        f"`matches.csv` and `deliveries.csv` into `data/raw/`."
    )

if data_loaded:
    n_matches = len(matches_df)
    n_seasons = matches_df["season"].nunique()
    n_teams = len(set(matches_df["team1"]) | set(matches_df["team2"]))
    n_venues = matches_df["venue"].nunique()
    n_players = len(players_df)
    season_min, season_max = int(matches_df["season"].min()), int(matches_df["season"].max())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches", f"{n_matches:,}")
    col2.metric("Seasons", f"{n_seasons}", help=f"{season_min}–{season_max}")
    col3.metric("Teams", f"{n_teams}")
    col4.metric("Venues", f"{n_venues}")
    col5.metric("Players", f"{n_players:,}")

    st.divider()

    st.subheader("🤖 Prediction Model Status")
    try:
        metadata = load_model_metadata()
        best_name = metadata["best_model_name"]
        best_metrics = metadata["results"][best_name]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Best Model", best_name)
        mc2.metric("Test Accuracy", f"{best_metrics['test_accuracy']:.1%}")
        mc3.metric("Test ROC-AUC", f"{best_metrics['test_roc_auc']:.3f}")
        mc4.metric("Test F1", f"{best_metrics['test_f1']:.3f}")
        st.caption(
            f"Trained on {metadata['train_rows']:,} matches, evaluated on "
            f"{metadata['test_rows']:,} held-out matches. See the "
            f"**Model Performance** page for the full comparison."
        )
    except ModelNotTrainedError:
        st.warning(
            "No trained model found yet. Run `python -m src.models.train` "
            "from the project root, then reload this page to see model "
            "status here and enable the **Match Prediction** page."
        )

    st.divider()

    st.subheader("📖 What's in this dashboard")
    st.markdown(
        """
- **🔮 Match Prediction** — predict the winner of a hypothetical matchup between two teams, with a win-probability breakdown.
- **🏟️ Team Analytics** — career win rates, head-to-head records, toss impact, venue performance, and season-by-season trends for any team.
- **🏏 Player Analytics** — batting and bowling leaderboards, qualified strike-rate/economy leaders, and individual player profiles.
- **📊 Match Insights** — league-wide scoring trends, venue characteristics (chasing vs. defending grounds), and season summaries.
- **📈 Model Performance** — side-by-side comparison of every trained model (accuracy, F1, ROC-AUC, confusion matrices, feature importance).
- **ℹ️ About** — dataset provenance, methodology notes, and known limitations.

Use the sidebar to navigate between pages.
        """
    )
