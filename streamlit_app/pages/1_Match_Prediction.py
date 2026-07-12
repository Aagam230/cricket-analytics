"""
1_Match_Prediction.py
======================
Streamlit page: predict the winner of a hypothetical IPL match.

Wraps `src.models.predict.predict_match()` with a simple form UI. All
prediction logic lives in `src/models/predict.py` -- this page is
presentation-only, per the project's separation-of-concerns design.
"""

import _bootstrap  # noqa: F401

from datetime import datetime

import streamlit as st

from data_service import get_matches
from src.models.predict import (
    ModelNotTrainedError,
    get_known_teams,
    get_known_venues,
    predict_match,
)
from src.utils.config import APP_ICON, APP_TITLE
from src.visualization.plots import plot_win_probability_gauge

st.set_page_config(page_title=f"Match Prediction | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("🔮 Match Prediction")
st.caption("Predict the winner of a hypothetical matchup between two IPL teams.")

try:
    teams = get_known_teams()
    venues_df = get_known_venues()
except ModelNotTrainedError as exc:
    st.warning(f"{exc}")
    st.stop()
except FileNotFoundError as exc:
    st.error(f"Could not load the IPL dataset: {exc}")
    st.stop()

with st.form("match_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with col2:
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", team2_options, index=0)

    venue = st.selectbox("Venue", venues_df["venue"].tolist())
    city = venues_df.loc[venues_df["venue"] == venue, "city"].iloc[0]
    st.caption(f"City: {city}")

    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.radio("Toss Winner", [team1, team2], horizontal=True)
    with col4:
        toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

    col5, col6 = st.columns(2)
    with col5:
        season = st.number_input(
            "Season", min_value=2008, max_value=2035, value=datetime.now().year, step=1
        )
    with col6:
        match_month = st.selectbox(
            "Match Month", list(range(1, 13)), index=datetime.now().month - 1,
            format_func=lambda m: datetime(2000, m, 1).strftime("%B"),
        )

    submitted = st.form_submit_button("Predict Winner", use_container_width=True, type="primary")

if submitted:
    try:
        result = predict_match(
            team1=team1,
            team2=team2,
            venue=venue,
            city=city,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            season=int(season),
            match_month=int(match_month),
        )
    except ModelNotTrainedError as exc:
        st.warning(f"{exc}")
        st.stop()
    except ValueError as exc:
        st.error(f"{exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction Result")

    winner = result["predicted_winner"]
    st.success(f"🏆 **{winner}** is predicted to win ({result['win_probability']:.1%} confidence)")

    fig = plot_win_probability_gauge(team1, team2, result["team1_win_probability"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Model used: {result['model_name']}")

    with st.expander("Show engineered features used for this prediction"):
        st.dataframe(result["features_used"], use_container_width=True)
