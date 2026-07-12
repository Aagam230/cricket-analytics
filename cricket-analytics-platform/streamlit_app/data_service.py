"""
data_service.py
================
Shared, Streamlit-cached data-loading helpers used by every page in the
dashboard.

Why this module exists:
    `src/data/preprocess.py::preprocess_all()` re-reads and re-cleans the
    raw CSVs (and re-derives players/seasons) from scratch every time it's
    called -- reasonable for a one-off script, but far too slow to call on
    every Streamlit page load / widget interaction. This module wraps it
    (and the model-loading calls in `src/models/predict.py`) with
    `st.cache_data` / `st.cache_resource` so the expensive work happens
    once per session, not once per click.

Every page imports from HERE rather than calling
`src.data.preprocess.preprocess_all()` directly, so caching behavior is
consistent across the whole app.
"""

import streamlit as st

from src.data.preprocess import preprocess_all


@st.cache_data(show_spinner="Loading and cleaning IPL data...")
def load_cleaned_data() -> dict:
    """
    Load and clean matches/deliveries/players/seasons once per session.

    Returns:
        Dict with keys "matches", "deliveries", "players", "seasons",
        each a cleaned pandas DataFrame (see `src/data/preprocess.py`).
        Cached by Streamlit -- subsequent calls within the same session
        return the cached result instantly instead of re-running the
        cleaning pipeline.
    """
    return preprocess_all(save=False)


def get_matches():
    """Convenience accessor: cleaned matches DataFrame."""
    return load_cleaned_data()["matches"]


def get_deliveries():
    """Convenience accessor: cleaned deliveries DataFrame."""
    return load_cleaned_data()["deliveries"]


def get_players():
    """Convenience accessor: derived players DataFrame."""
    return load_cleaned_data()["players"]


def get_seasons():
    """Convenience accessor: derived seasons DataFrame."""
    return load_cleaned_data()["seasons"]
