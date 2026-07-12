"""
7_Live_Win_Probability.py
==========================
Streamlit page: the IN-MATCH ("live") win-probability model -- the
counterpart to page 1 (Match Prediction, which predicts a winner BEFORE
the match starts). See `docs/MODEL_CARD.md` "Part 2" for the full
methodology and honest performance numbers.

Two modes:
    1. Match Replay -- pick a real historical chase and watch the model's
       win-probability curve for the chasing team ball-by-ball, with
       wicket-fall moments marked.
    2. Manual Calculator -- enter an arbitrary match state (target, score,
       wickets, overs) and get an instant win-probability estimate.
"""

import _bootstrap  # noqa: F401

import streamlit as st

from data_service import get_deliveries, get_matches
from src.models.live_win_probability import (
    LiveModelNotTrainedError,
    compute_match_win_probability_timeline,
    load_live_model_metadata,
    predict_live_win_probability,
)
from src.utils.config import APP_ICON, APP_TITLE
from src.visualization.plots import plot_win_probability_timeline

st.set_page_config(page_title=f"Live Win Probability | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("⚡ Live Win Probability")
st.caption(
    "Predicts the chasing team's win probability from the current match "
    "state (score, wickets, overs), rather than pre-match information. "
    "See the **About** page and `docs/MODEL_CARD.md` for why this model "
    "performs substantially better than pre-match prediction."
)

try:
    metadata = load_live_model_metadata()
except LiveModelNotTrainedError as exc:
    st.warning(f"{exc}")
    st.stop()

best = metadata["results"][metadata["best_model_name"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", metadata["best_model_name"])
c2.metric("Test Accuracy", f"{best['test_accuracy']:.1%}")
c3.metric("Test ROC-AUC", f"{best['test_roc_auc']:.3f}")
c4.metric("Trained on", f"{metadata['train_matches']:,} matches")

try:
    matches_df = get_matches()
    deliveries_df = get_deliveries()
except FileNotFoundError as exc:
    st.error(f"Could not load the IPL dataset: {exc}")
    st.stop()

tab_replay, tab_calculator = st.tabs(["📺 Match Replay", "🧮 Manual Calculator"])

with tab_replay:
    st.subheader("Replay a Real Chase")

    eligible = matches_df[
        (~matches_df["is_no_result"])
        & matches_df["first_innings_score"].notna()
        & (matches_df["method"] != "D/L")
    ].sort_values(["season", "date"], ascending=[False, False])

    eligible = eligible.copy()
    eligible["label"] = (
        eligible["season"].astype(str) + " — " + eligible["team1"] + " vs " + eligible["team2"]
        + " (" + eligible["venue"] + ")"
    )
    selected_label = st.selectbox("Select a match", eligible["label"].tolist())
    selected_match_id = eligible.loc[eligible["label"] == selected_label, "match_id"].iloc[0]
    match_row = eligible[eligible["match_id"] == selected_match_id].iloc[0]

    # Batting/bowling teams in the 2nd innings are read directly off the
    # timeline data below (safer than inferring from toss_decision, which
    # doesn't always map cleanly to who ends up batting second).
    timeline = compute_match_win_probability_timeline(selected_match_id, matches_df=matches_df, deliveries_df=deliveries_df)

    if timeline is None or timeline.empty:
        st.info("No ball-by-ball second-innings data available for this match.")
    else:
        batting_team = timeline["batting_team"].iloc[0]
        bowling_team = timeline["bowling_team"].iloc[0]
        target = int(match_row["first_innings_score"]) + 1

        result_text = f"🏆 {match_row['winner']} won"
        if match_row["win_by"] == match_row["win_by"] and match_row["win_margin"] == match_row["win_margin"]:
            result_text += f" by {int(match_row['win_margin'])} {match_row['win_by']}"
        st.caption(result_text)

        fig = plot_win_probability_timeline(timeline, batting_team, bowling_team, target)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"{batting_team} needed {target} to win. Each dip/rise reflects "
            f"the model re-evaluating win probability after every ball; "
            f"✕ markers indicate a wicket falling."
        )

with tab_calculator:
    st.subheader("Manual Win Probability Calculator")
    st.caption("Enter any hypothetical mid-chase situation to get an instant win-probability estimate.")

    all_teams = sorted(set(matches_df["team1"]) | set(matches_df["team2"]))
    all_venues = sorted(matches_df["venue"].unique())

    with st.form("live_calculator_form"):
        col1, col2 = st.columns(2)
        with col1:
            batting_team_input = st.selectbox("Batting (chasing) team", all_teams, index=0)
        with col2:
            bowling_options = [t for t in all_teams if t != batting_team_input]
            bowling_team_input = st.selectbox("Bowling (defending) team", bowling_options, index=0)

        venue_input = st.selectbox("Venue", all_venues)
        target_input = st.number_input("Target (runs to win)", min_value=1, max_value=350, value=180, step=1)

        col3, col4, col5 = st.columns(3)
        with col3:
            runs_scored_input = st.number_input("Current score", min_value=0, max_value=350, value=90, step=1)
        with col4:
            wickets_input = st.number_input("Wickets lost", min_value=0, max_value=10, value=3, step=1)
        with col5:
            overs_bowled_input = st.number_input("Overs completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

        submitted = st.form_submit_button("Calculate Win Probability", use_container_width=True, type="primary")

    if submitted:
        balls_bowled_input = int(round(overs_bowled_input * 6))
        try:
            result = predict_live_win_probability(
                batting_team=batting_team_input,
                bowling_team=bowling_team_input,
                venue=venue_input,
                target=int(target_input),
                runs_scored=int(runs_scored_input),
                wickets_lost=int(wickets_input),
                balls_bowled=balls_bowled_input,
            )
        except LiveModelNotTrainedError as exc:
            st.warning(f"{exc}")
            st.stop()
        except ValueError as exc:
            st.error(f"{exc}")
            st.stop()

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{batting_team_input} Win Probability", f"{result['batting_team_win_probability']:.1%}")
        m2.metric("Runs Needed", f"{result['runs_needed']}")
        m3.metric("Required Run Rate", f"{result['required_run_rate']:.2f}")

        st.progress(result["batting_team_win_probability"])
        st.caption(f"Model used: {result['model_name']}")
