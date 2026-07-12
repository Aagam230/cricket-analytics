"""
4_Match_Insights.py
====================
Streamlit page: league-wide match insights -- scoring trends over time,
venue characteristics (chasing vs. defending grounds), and season
summaries.

All computation is delegated to `src/analytics/venue_analytics.py` and
`src/data/preprocess.py::derive_seasons()` (via `data_service.get_seasons()`).
"""

import _bootstrap  # noqa: F401

import streamlit as st

from data_service import get_matches, get_seasons
from src.analytics.venue_analytics import compute_city_summary, compute_venue_profile, compute_venue_summary
from src.utils.config import APP_ICON, APP_TITLE
from src.visualization.plots import (
    plot_season_runs_trend,
    plot_venue_bat_first_win_pct,
    plot_venue_score_comparison,
)

st.set_page_config(page_title=f"Match Insights | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("📊 Match Insights")

try:
    matches_df = get_matches()
    seasons_df = get_seasons()
except FileNotFoundError as exc:
    st.error(f"Could not load the IPL dataset: {exc}")
    st.stop()

tab_scoring, tab_venues, tab_seasons = st.tabs(["Scoring Trends", "Venue Analysis", "Season Summaries"])

with tab_scoring:
    st.subheader("League-Wide Scoring Trend")
    st.plotly_chart(plot_season_runs_trend(matches_df), use_container_width=True)

    st.caption(
        "This dataset has no day/night or match-start-time field, so a "
        "day vs. night scoring comparison (present in earlier versions of "
        "this dashboard) isn't available here."
    )

with tab_venues:
    venue_summary = compute_venue_summary(matches_df)
    top_n = st.slider("Show top N venues (by matches played)", min_value=5, max_value=min(30, len(venue_summary)), value=12)

    st.plotly_chart(plot_venue_score_comparison(venue_summary, top_n=top_n), use_container_width=True)
    st.plotly_chart(plot_venue_bat_first_win_pct(venue_summary, top_n=top_n), use_container_width=True)

    st.subheader("Venue Deep-Dive")
    venue = st.selectbox("Select a venue", venue_summary["venue"].tolist())
    profile = compute_venue_profile(matches_df, venue, venue_summary=venue_summary)

    if profile["summary"] is not None:
        s = profile["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches Played", int(s["matches_played"]))
        c2.metric("Avg 1st Innings", f"{s['avg_first_innings_score']:.0f}" if s["avg_first_innings_score"] == s["avg_first_innings_score"] else "—")
        c3.metric("Avg 2nd Innings", f"{s['avg_second_innings_score']:.0f}" if s["avg_second_innings_score"] == s["avg_second_innings_score"] else "—")
        c4.metric("Bat-First Win %", f"{s['bat_first_win_pct']:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Most frequent teams at this venue")
        st.dataframe(profile["most_frequent_teams"], use_container_width=True, hide_index=True)
    with col2:
        st.caption("Highest-scoring matches here")
        st.dataframe(profile["highest_scoring_matches"], use_container_width=True, hide_index=True)

    st.subheader("City Summary")
    city_summary = compute_city_summary(matches_df)
    st.dataframe(city_summary, use_container_width=True, hide_index=True)

with tab_seasons:
    st.subheader("Season-by-Season Summary")
    st.caption(
        "Champion/runner-up are inferred as the winner/loser of each "
        "season's chronologically last match, since this dataset has no "
        "explicit 'this was the final' flag."
    )
    st.dataframe(seasons_df, use_container_width=True, hide_index=True)

    season_options = seasons_df["season"].tolist()
    selected_season = st.selectbox("Select a season for details", season_options, index=len(season_options) - 1)
    season_row = seasons_df[seasons_df["season"] == selected_season].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion", season_row["champion"] if season_row["champion"] == season_row["champion"] else "—")
    c2.metric("Runner-Up", season_row["runner_up"] if season_row["runner_up"] == season_row["runner_up"] else "—")
    c3.metric("Orange Cap", season_row["orange_cap_winner"] if season_row["orange_cap_winner"] == season_row["orange_cap_winner"] else "—")
    c4.metric("Purple Cap", season_row["purple_cap_winner"] if season_row["purple_cap_winner"] == season_row["purple_cap_winner"] else "—")
