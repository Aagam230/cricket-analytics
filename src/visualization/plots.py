"""
plots.py
========
Shared Plotly chart-building functions for the Cricket Analytics & Match
Prediction Platform's Streamlit dashboard.

This module is the ONLY place that imports `plotly.graph_objects` /
`plotly.express` for chart construction. Every Streamlit page calls into
here rather than building figures inline, so:
    1. The dashboard has a single consistent visual style (colors, fonts,
       hover behavior) defined once.
    2. Chart logic is unit-testable independent of Streamlit.
    3. Adding a new page that needs "win pct by team" or "ROC curve" never
       requires re-deriving the Plotly boilerplate.

Usage:
    from src.visualization.plots import plot_team_win_pct_bar

    fig = plot_team_win_pct_bar(team_summary_df)
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette used across every chart in the dashboard.
PRIMARY_COLOR: str = "#1B4F72"
SECONDARY_COLOR: str = "#CB4335"
ACCENT_COLOR: str = "#F4D03F"
SEQUENTIAL_PALETTE: list[str] = px.colors.sequential.Tealgrn
CATEGORICAL_PALETTE: list[str] = px.colors.qualitative.Bold

PLOT_TEMPLATE: str = "plotly_white"


def _apply_standard_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """
    Apply consistent title styling, template, margins, and height to any
    figure built in this module, so every chart in the dashboard looks
    like part of the same product rather than a patchwork of defaults.

    Args:
        fig: A Plotly Figure to style in place.
        title: Chart title text.
        height: Chart height in pixels.

    Returns:
        The same Figure object, with layout updated (also mutated in place).
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_team_win_pct_bar(team_summary_df: pd.DataFrame, top_n: int | None = None) -> go.Figure:
    """
    Horizontal bar chart of win percentage by team.

    Args:
        team_summary_df: Output of `team_analytics.compute_team_summary()`.
        top_n: If given, show only the top N teams by win_pct.

    Returns:
        Plotly Figure.
    """
    df = team_summary_df.copy()
    if top_n is not None:
        df = df.head(top_n)
    df = df.sort_values("win_pct")

    fig = px.bar(
        df,
        x="win_pct",
        y="team",
        orientation="h",
        text="win_pct",
        color="win_pct",
        color_continuous_scale=SEQUENTIAL_PALETTE,
        labels={"win_pct": "Win %", "team": "Team"},
        hover_data={"matches_played": True, "wins": True, "losses": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, "Team Win Percentage (Career)")


def plot_season_trend_line(season_perf_df: pd.DataFrame, team_name: str) -> go.Figure:
    """
    Line chart of a single team's win percentage across IPL seasons.

    Args:
        season_perf_df: Output of `team_analytics.compute_season_wise_performance()`.
        team_name: Team name, used in the chart title.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        season_perf_df,
        x="season",
        y="win_pct",
        markers=True,
        labels={"season": "Season", "win_pct": "Win %"},
        hover_data={"matches_played": True, "wins": True},
    )
    fig.update_traces(line_color=PRIMARY_COLOR, marker=dict(size=8, color=SECONDARY_COLOR))
    fig.update_yaxes(range=[0, 100])
    return _apply_standard_layout(fig, f"{team_name} -- Win % by Season")


def plot_head_to_head_pie(team_a: str, team_b: str, h2h: dict) -> go.Figure:
    """
    Pie/donut chart summarizing a head-to-head record between two teams.

    Args:
        team_a: First team's name.
        team_b: Second team's name.
        h2h: Output dict of `team_analytics.compute_head_to_head()`.

    Returns:
        Plotly Figure.
    """
    labels = [f"{team_a} wins", f"{team_b} wins"]
    values = [h2h["team_a_wins"], h2h["team_b_wins"]]
    if h2h["no_results"] > 0:
        labels.append("No result / tie")
        values.append(h2h["no_results"])

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.45, marker=dict(colors=CATEGORICAL_PALETTE))]
    )
    return _apply_standard_layout(fig, f"{team_a} vs {team_b} -- Head to Head ({h2h['total_matches']} matches)")


def plot_toss_impact_bar(toss_impact: dict) -> go.Figure:
    """
    Bar chart showing how often the toss winner also wins the match,
    broken down by what they chose to do (bat or field first).

    Args:
        toss_impact: Output dict of `team_analytics.compute_toss_impact()`.

    Returns:
        Plotly Figure.
    """
    df = toss_impact["by_decision"]
    fig = px.bar(
        df,
        x="toss_decision",
        y="toss_winner_also_match_winner_pct",
        text="toss_winner_also_match_winner_pct",
        color="toss_decision",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={
            "toss_decision": "Toss Decision",
            "toss_winner_also_match_winner_pct": "Toss Winner Also Won Match (%)",
        },
        hover_data={"matches": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100])
    fig.add_hline(
        y=50, line_dash="dash", line_color="gray",
        annotation_text="50% (no advantage)", annotation_position="top left",
    )
    return _apply_standard_layout(fig, "Does Winning the Toss Help You Win the Match?")


def plot_venue_score_comparison(venue_summary_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Grouped bar chart comparing average first- vs second-innings scores
    across the most-used venues.

    Args:
        venue_summary_df: Output of `venue_analytics.compute_venue_summary()`.
        top_n: Number of venues (by matches played) to include.

    Returns:
        Plotly Figure.
    """
    df = venue_summary_df.head(top_n)
    fig = go.Figure()
    fig.add_bar(name="Avg 1st Innings Score", x=df["venue"], y=df["avg_first_innings_score"], marker_color=PRIMARY_COLOR)
    fig.add_bar(name="Avg 2nd Innings Score", x=df["venue"], y=df["avg_second_innings_score"], marker_color=SECONDARY_COLOR)
    fig.update_layout(barmode="group", xaxis_tickangle=-35)
    return _apply_standard_layout(fig, "Average Innings Scores by Venue", height=480)


def plot_venue_bat_first_win_pct(venue_summary_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Bar chart of how often the team batting first wins, by venue -- a
    quick visual proxy for "chasing ground" vs "defending ground".

    Args:
        venue_summary_df: Output of `venue_analytics.compute_venue_summary()`.
        top_n: Number of venues (by matches played) to include.

    Returns:
        Plotly Figure.
    """
    df = venue_summary_df.head(top_n).sort_values("bat_first_win_pct")
    fig = px.bar(
        df,
        x="bat_first_win_pct",
        y="venue",
        orientation="h",
        text="bat_first_win_pct",
        color="bat_first_win_pct",
        color_continuous_scale=SEQUENTIAL_PALETTE,
        labels={"bat_first_win_pct": "Bat-First Win %", "venue": "Venue"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_vline(x=50, line_dash="dash", line_color="gray")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, "Bat-First Win % by Venue (Chasing vs Defending Grounds)", height=480)


def plot_batting_leaderboard_bar(batting_df: pd.DataFrame, metric: str = "runs", top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of the top N batters by a chosen metric.

    Args:
        batting_df: Output of `player_analytics.compute_batting_leaderboard()`.
        metric: Column to rank/plot by -- typically "runs" or "strike_rate".
        top_n: Number of players to show.

    Returns:
        Plotly Figure.
    """
    df = batting_df.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    label = metric.replace("_", " ").title()
    fig = px.bar(
        df,
        x=metric,
        y="player",
        orientation="h",
        text=metric,
        color=metric,
        color_continuous_scale=SEQUENTIAL_PALETTE,
        labels={metric: label, "player": "Player"},
        hover_data={"innings": True, "average": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, f"Top {top_n} Batters by {label}")


def plot_bowling_leaderboard_bar(bowling_df: pd.DataFrame, metric: str = "wickets", top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of the top N bowlers by a chosen metric.

    Args:
        bowling_df: Output of `player_analytics.compute_bowling_leaderboard()`.
        metric: Column to rank/plot by -- typically "wickets" or "economy"
            (note: for "economy", lower is better, so callers typically
            want to pre-sort/filter for qualified bowlers before calling).
        top_n: Number of players to show.

    Returns:
        Plotly Figure.
    """
    ascending = metric == "economy"
    df = bowling_df.sort_values(metric, ascending=ascending).head(top_n).sort_values(metric, ascending=not ascending)
    label = metric.replace("_", " ").title()
    fig = px.bar(
        df,
        x=metric,
        y="player",
        orientation="h",
        text=metric,
        color=metric,
        color_continuous_scale=SEQUENTIAL_PALETTE if not ascending else SEQUENTIAL_PALETTE[::-1],
        labels={metric: label, "player": "Player"},
        hover_data={"innings": True, "overs": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, f"Top {top_n} Bowlers by {label}")


def plot_model_comparison_bar(results: dict, metric: str = "cv_f1_mean") -> go.Figure:
    """
    Bar chart comparing every trained model on a single metric, used on
    the "Model Performance" Streamlit page.

    Args:
        results: The "results" dict from `train.train_and_compare_models()`
            (or the equivalent section of `model_metadata.json`), keyed by
            model name.
        metric: Which metric key to plot (e.g. "cv_f1_mean", "test_accuracy",
            "test_roc_auc").

    Returns:
        Plotly Figure.
    """
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    df = pd.DataFrame({"model": names, "value": values}).sort_values("value")

    label = metric.replace("_", " ").title()
    fig = px.bar(
        df,
        x="value",
        y="model",
        orientation="h",
        text="value",
        color="value",
        color_continuous_scale=SEQUENTIAL_PALETTE,
        labels={"value": label, "model": "Model"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, f"Model Comparison -- {label}")


def plot_multi_metric_comparison(results: dict, metrics: list[str] | None = None) -> go.Figure:
    """
    Grouped bar chart comparing every trained model across several test-set
    metrics at once (accuracy, precision, recall, F1, ROC-AUC).

    Args:
        results: The "results" dict from `train.train_and_compare_models()`.
        metrics: List of metric keys to include. Defaults to the standard
            five test-set metrics.

    Returns:
        Plotly Figure.
    """
    if metrics is None:
        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]

    fig = go.Figure()
    for i, model_name in enumerate(results.keys()):
        values = [results[model_name][m] for m in metrics]
        fig.add_bar(
            name=model_name,
            x=[m.replace("test_", "").replace("_", " ").title() for m in metrics],
            y=values,
            marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
        )
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]))
    return _apply_standard_layout(fig, "Model Comparison -- All Test-Set Metrics", height=460)


def plot_confusion_matrix_heatmap(cm: list[list[int]], model_name: str = "") -> go.Figure:
    """
    Heatmap of a 2x2 confusion matrix with annotated cell counts.

    Args:
        cm: Confusion matrix as a 2x2 nested list, [[TN, FP], [FN, TP]]
            (the format returned by `evaluate.evaluate_model()`).
        model_name: Optional model name to include in the chart title.

    Returns:
        Plotly Figure.
    """
    cm_array = np.array(cm)
    labels = ["Team 2 Wins", "Team 1 Wins"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_array,
            x=[f"Predicted: {l}" for l in labels],
            y=[f"Actual: {l}" for l in labels],
            text=cm_array,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale=SEQUENTIAL_PALETTE,
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    title = f"Confusion Matrix -- {model_name}" if model_name else "Confusion Matrix"
    return _apply_standard_layout(fig, title, height=420)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, model_name: str = "") -> go.Figure:
    """
    ROC curve for a single model, with the diagonal "random guess"
    reference line and the AUC shown in the title.

    Args:
        fpr: False-positive-rate array (from `evaluate.get_roc_curve_points()`).
        tpr: True-positive-rate array.
        auc: Area under the ROC curve.
        model_name: Optional model name to include in the chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve", line=dict(color=PRIMARY_COLOR, width=3)))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color="gray", dash="dash"))
    )
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate", range=[0, 1])
    title = f"ROC Curve -- {model_name} (AUC = {auc:.3f})" if model_name else f"ROC Curve (AUC = {auc:.3f})"
    return _apply_standard_layout(fig, title)


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of the top N most important features for a
    fitted model.

    Args:
        importance_df: Output of `evaluate.get_feature_importance()`.
        top_n: Number of features to show.

    Returns:
        Plotly Figure.
    """
    df = importance_df.head(top_n).sort_values("importance")
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        color="importance",
        color_continuous_scale=SEQUENTIAL_PALETTE,
        labels={"importance": "Relative Importance", "feature": "Feature"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_standard_layout(fig, "Feature Importance", height=480)


def plot_win_probability_gauge(team1: str, team2: str, team1_prob: float) -> go.Figure:
    """
    A horizontal stacked "tug of war" bar showing each team's predicted
    win probability for the Match Prediction page -- more intuitive for a
    head-to-head matchup than a single gauge.

    Args:
        team1: First team's name.
        team2: Second team's name.
        team1_prob: Predicted probability (0-1) that team1 wins.

    Returns:
        Plotly Figure.
    """
    team2_prob = 1 - team1_prob
    fig = go.Figure()
    fig.add_bar(
        y=["Win Probability"], x=[team1_prob * 100], name=team1,
        orientation="h", marker_color=PRIMARY_COLOR,
        text=f"{team1}: {team1_prob:.1%}", textposition="inside",
    )
    fig.add_bar(
        y=["Win Probability"], x=[team2_prob * 100], name=team2,
        orientation="h", marker_color=SECONDARY_COLOR,
        text=f"{team2}: {team2_prob:.1%}", textposition="inside",
    )
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], title="Win Probability (%)"), showlegend=False)
    return _apply_standard_layout(fig, "Predicted Win Probability", height=200)


def plot_season_runs_trend(matches_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of average first-innings score by season across the whole
    league, useful for visualizing scoring-rate trends over IPL history
    (e.g. the impact of rule changes, bat technology, shorter boundaries).

    Args:
        matches_df: Cleaned matches DataFrame.

    Returns:
        Plotly Figure.
    """
    season_avg = (
        matches_df.groupby("season")["first_innings_score"]
        .mean()
        .round(1)
        .reset_index(name="avg_first_innings_score")
    )
    fig = px.line(
        season_avg,
        x="season",
        y="avg_first_innings_score",
        markers=True,
        labels={"season": "Season", "avg_first_innings_score": "Avg 1st Innings Score"},
    )
    fig.update_traces(line_color=PRIMARY_COLOR, marker=dict(size=8, color=ACCENT_COLOR, line=dict(width=1, color=PRIMARY_COLOR)))
    return _apply_standard_layout(fig, "League-Wide Scoring Trend by Season")
