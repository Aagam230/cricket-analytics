"""
2_Team_Analytics.py
====================
Streamlit page: team-level analytics -- career win rates, head-to-head
records, toss impact, venue performance, and season-by-season trends.

All computation is delegated to `src/analytics/team_analytics.py`; this
page only handles widget layout and chart rendering via
`src/visualization/plots.py`.
"""

import _bootstrap  # noqa: F401

import streamlit as st

from data_service import get_matches
from src.analytics.team_analytics import (
    compute_head_to_head,
    compute_recent_form,
    compute_season_wise_performance,
    compute_team_summary,
    compute_team_venue_performance,
    compute_toss_impact,
)
from src.utils.config import APP_ICON, APP_TITLE
from src.visualization.plots import (
    plot_head_to_head_pie,
    plot_season_trend_line,
    plot_team_win_pct_bar,
    plot_toss_impact_bar,
)

st.set_page_config(page_title=f"Team Analytics | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("🏟️ Team Analytics")

try:
    matches_df = get_matches()
except FileNotFoundError as exc:
    st.error(f"Could not load the IPL dataset: {exc}")
    st.stop()

team_summary = compute_team_summary(matches_df)
all_teams = sorted(set(matches_df["team1"]) | set(matches_df["team2"]))

tab_overview, tab_single_team, tab_h2h, tab_toss = st.tabs(
    ["League Overview", "Single Team Deep-Dive", "Head-to-Head", "Toss Impact"]
)

with tab_overview:
    st.subheader("Career Win Percentage by Team")
    top_n = st.slider("Show top N teams", min_value=3, max_value=len(team_summary), value=len(team_summary))
    st.plotly_chart(plot_team_win_pct_bar(team_summary, top_n=top_n), use_container_width=True)
    st.dataframe(team_summary, use_container_width=True, hide_index=True)

with tab_single_team:
    team = st.selectbox("Select a team", all_teams, key="single_team_select")

    row = team_summary[team_summary["team"] == team]
    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches Played", int(r["matches_played"]))
        c2.metric("Wins", int(r["wins"]))
        c3.metric("Losses", int(r["losses"]))
        c4.metric("Win %", f"{r['win_pct']:.1f}%")

    recent = compute_recent_form(matches_df, team, last_n=5)
    st.caption(
        f"Recent form (last {recent['matches_played']} matches): "
        f"{recent['wins']} wins ({recent['win_pct']:.1f}%)"
    )

    season_perf = compute_season_wise_performance(matches_df, team)
    if not season_perf.empty:
        st.plotly_chart(plot_season_trend_line(season_perf, team), use_container_width=True)
    else:
        st.info("No season-wise data available for this team.")

    st.subheader(f"{team} — Performance by Venue")
    venue_perf = compute_team_venue_performance(matches_df, team)
    st.dataframe(venue_perf, use_container_width=True, hide_index=True)

with tab_h2h:
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", all_teams, index=0, key="h2h_team_a")
    with col2:
        team_b_options = [t for t in all_teams if t != team_a]
        team_b = st.selectbox("Team B", team_b_options, index=0, key="h2h_team_b")

    h2h = compute_head_to_head(matches_df, team_a, team_b)
    if h2h["total_matches"] == 0:
        st.info(f"{team_a} and {team_b} have never played each other in this dataset.")
    else:
        st.plotly_chart(plot_head_to_head_pie(team_a, team_b, h2h), use_container_width=True)
        with st.expander("Show all head-to-head matches"):
            st.dataframe(h2h["matches"], use_container_width=True, hide_index=True)

with tab_toss:
    toss_impact = compute_toss_impact(matches_df)
    st.metric(
        "Toss winner also wins the match",
        f"{toss_impact['overall_toss_win_match_win_pct']:.1f}%",
        help="Percentage of decided matches where the toss winner went on to win.",
    )
    st.plotly_chart(plot_toss_impact_bar(toss_impact), use_container_width=True)
