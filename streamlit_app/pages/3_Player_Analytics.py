"""
3_Player_Analytics.py
======================
Streamlit page: player-level analytics -- batting/bowling leaderboards,
qualified strike-rate/economy leaders, and individual player profiles.

All computation is delegated to `src/analytics/player_analytics.py`.

NOTE on player metadata: this dataset has no raw players.csv, so the
`players_clean.csv` used here is DERIVED purely from ball-by-ball
appearances (see `src/data/preprocess.py::derive_players()`). Fields like
nationality, batting/bowling style, playing role, and auction price are
not available -- the profile section below only shows what's actually
derivable (debut/last season, matches played, career runs/wickets).
"""

import _bootstrap  # noqa: F401

import streamlit as st

from data_service import get_deliveries, get_matches, get_players
from src.analytics.player_analytics import (
    compute_batting_leaderboard,
    compute_bowling_leaderboard,
    compute_player_profile,
    compute_player_season_timeline,
    compute_qualified_economy_leaderboard,
    compute_qualified_strike_rate_leaderboard,
)
from src.utils.config import APP_ICON, APP_TITLE
from src.visualization.plots import plot_batting_leaderboard_bar, plot_bowling_leaderboard_bar

st.set_page_config(page_title=f"Player Analytics | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("🏏 Player Analytics")

try:
    deliveries_df = get_deliveries()
    matches_df = get_matches()
    players_df = get_players()
except FileNotFoundError as exc:
    st.error(f"Could not load the IPL dataset: {exc}")
    st.stop()

st.info(
    "This dataset has no player-metadata file (nationality, batting/bowling "
    "style, playing role, auction price aren't available) -- the profile "
    "below shows only what can be derived from ball-by-ball appearances.",
    icon="ℹ️",
)

tab_batting, tab_bowling, tab_profile = st.tabs(["Batting Leaders", "Bowling Leaders", "Player Profile"])

with tab_batting:
    top_n = st.slider("Show top N batters", min_value=5, max_value=30, value=10, key="bat_top_n")
    batting_lb = compute_batting_leaderboard(deliveries_df, top_n=top_n)

    metric = st.selectbox(
        "Rank by", ["runs", "strike_rate", "average"], format_func=str.title, key="bat_metric"
    )
    if metric == "runs":
        st.plotly_chart(plot_batting_leaderboard_bar(batting_lb, metric="runs", top_n=top_n), use_container_width=True)
    else:
        qualified = compute_qualified_strike_rate_leaderboard(deliveries_df, top_n=top_n)
        st.dataframe(qualified, use_container_width=True, hide_index=True)

    st.dataframe(batting_lb, use_container_width=True, hide_index=True)

with tab_bowling:
    top_n_b = st.slider("Show top N bowlers", min_value=5, max_value=30, value=10, key="bowl_top_n")
    bowling_lb = compute_bowling_leaderboard(deliveries_df, top_n=top_n_b)

    metric_b = st.selectbox(
        "Rank by", ["wickets", "economy"], format_func=str.title, key="bowl_metric"
    )
    if metric_b == "wickets":
        st.plotly_chart(plot_bowling_leaderboard_bar(bowling_lb, metric="wickets", top_n=top_n_b), use_container_width=True)
    else:
        qualified_econ = compute_qualified_economy_leaderboard(deliveries_df, top_n=top_n_b)
        st.dataframe(qualified_econ, use_container_width=True, hide_index=True)

    st.dataframe(bowling_lb, use_container_width=True, hide_index=True)

with tab_profile:
    all_players = sorted(players_df["player_name"].unique())
    player_name = st.selectbox("Select a player", all_players)

    profile = compute_player_profile(deliveries_df, players_df, player_name)

    if profile["metadata"] is not None:
        meta = profile["metadata"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Debut Season", int(meta["ipl_debut_season"]) if meta["ipl_debut_season"] == meta["ipl_debut_season"] else "—")
        c2.metric("Last Season", int(meta["last_season_played"]) if meta["last_season_played"] == meta["last_season_played"] else "—")
        c3.metric("Matches Batted", int(meta["matches_batted"]))
        c4.metric("Matches Bowled", int(meta["matches_bowled"]))

    col_bat, col_bowl = st.columns(2)
    with col_bat:
        st.subheader("Batting")
        if profile["batting"] is not None:
            b = profile["batting"]
            st.metric("Career Runs", int(b["runs"]))
            st.metric("Strike Rate", f"{b['strike_rate']:.2f}")
            st.metric("Average", f"{b['average']:.2f}" if b["average"] == b["average"] else "—")
        else:
            st.caption("No batting record.")
    with col_bowl:
        st.subheader("Bowling")
        if profile["bowling"] is not None:
            bo = profile["bowling"]
            st.metric("Career Wickets", int(bo["wickets"]))
            st.metric("Economy", f"{bo['economy']:.2f}")
            st.metric("Average", f"{bo['average']:.2f}" if bo["average"] == bo["average"] else "—")
        else:
            st.caption("No bowling record.")

    st.subheader(f"{player_name} — Season-by-Season Timeline")
    timeline = compute_player_season_timeline(deliveries_df, matches_df, player_name)
    st.dataframe(timeline, use_container_width=True, hide_index=True)
