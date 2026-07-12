"""
5_Model_Performance.py
=======================
Streamlit page: side-by-side comparison of every trained model, plus a
detailed drill-down (confusion matrix, ROC curve, feature importance) for
any individual model.

Reads `models/model_metadata.json` (written by `src/models/train.py`) for
the comparison table, and re-evaluates a fitted model on-the-fly (loaded
from `models/all_models.joblib`) for the ROC curve / feature importance,
since those need the actual fitted estimator + test set, not just the
persisted summary metrics.
"""

import _bootstrap  # noqa: F401

import joblib
import streamlit as st

from src.features.engineering import build_model_dataset
from src.models.evaluate import get_feature_importance, get_roc_curve_points
from src.models.predict import ModelNotTrainedError, load_model_metadata
from src.utils.config import APP_ICON, APP_TITLE, MODELS_DIR, RANDOM_SEED, TEST_SIZE
from src.visualization.plots import (
    plot_confusion_matrix_heatmap,
    plot_feature_importance_bar,
    plot_model_comparison_bar,
    plot_multi_metric_comparison,
    plot_roc_curve,
)

st.set_page_config(page_title=f"Model Performance | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("📈 Model Performance")

try:
    metadata = load_model_metadata()
except ModelNotTrainedError as exc:
    st.warning(f"{exc}")
    st.stop()

results = metadata["results"]

split_kind = "time-based (test = most recent matches)" if metadata.get("time_based_split", True) else "random stratified"
st.caption(
    f"Trained {metadata['trained_at_utc']} · {metadata['train_rows']:,} train rows · "
    f"{metadata['test_rows']:,} test rows ({split_kind} split) · {metadata['cv_folds']}-fold CV · "
    f"selection metric: {metadata['primary_selection_metric']}"
)

st.subheader(f"🏆 Best Model: {metadata['best_model_name']}")

tab_compare, tab_drilldown = st.tabs(["Model Comparison", "Model Drill-Down"])

with tab_compare:
    metric_choice = st.selectbox(
        "Primary comparison metric",
        ["cv_f1_mean", "cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean", "cv_roc_auc_mean"],
        format_func=lambda m: m.replace("cv_", "CV ").replace("_mean", "").replace("_", " ").title(),
    )
    st.plotly_chart(plot_model_comparison_bar(results, metric=metric_choice), use_container_width=True)
    st.plotly_chart(plot_multi_metric_comparison(results), use_container_width=True)

    st.subheader("Full Metrics Table")
    st.dataframe(
        {name: {k: v for k, v in m.items() if k != "test_confusion_matrix"} for name, m in results.items()},
        use_container_width=True,
    )

with tab_drilldown:
    model_name = st.selectbox("Select a model", list(results.keys()))
    model_metrics = results[model_name]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Accuracy", f"{model_metrics['test_accuracy']:.3f}")
    c2.metric("Precision", f"{model_metrics['test_precision']:.3f}")
    c3.metric("Recall", f"{model_metrics['test_recall']:.3f}")
    c4.metric("F1", f"{model_metrics['test_f1']:.3f}")
    c5.metric("ROC-AUC", f"{model_metrics['test_roc_auc']:.3f}")

    st.plotly_chart(
        plot_confusion_matrix_heatmap(model_metrics["test_confusion_matrix"], model_name=model_name),
        use_container_width=True,
    )

    all_models_path = MODELS_DIR / "all_models.joblib"
    if all_models_path.exists():
        with st.spinner("Loading fitted model and rebuilding test set for ROC curve / feature importance..."):
            all_models = joblib.load(all_models_path)
            fitted_model = all_models.get(model_name)

            if fitted_model is not None:
                from sklearn.model_selection import train_test_split

                X, y, feature_names, _encoders = build_model_dataset(save=False)

                # Reconstruct the SAME train/test split used during
                # training (see metadata["time_based_split"]), or the ROC
                # curve / feature importance below would be computed on a
                # different test set than the one the persisted metrics
                # actually reflect.
                if metadata.get("time_based_split", True):
                    split_idx = int(len(X) * (1 - TEST_SIZE))
                    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
                else:
                    _X_train, X_test, _y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
                    )

                roc_points = get_roc_curve_points(fitted_model, X_test, y_test)
                st.plotly_chart(
                    plot_roc_curve(
                        roc_points["fpr"], roc_points["tpr"], model_metrics["test_roc_auc"], model_name=model_name
                    ),
                    use_container_width=True,
                )

                importance_df = get_feature_importance(fitted_model, feature_names)
                if importance_df is not None and not importance_df.empty:
                    st.plotly_chart(plot_feature_importance_bar(importance_df), use_container_width=True)
                else:
                    st.caption(f"{model_name} does not expose feature importances / coefficients.")
    else:
        st.info("Run `python -m src.models.train` to generate `all_models.joblib` for ROC curve / feature importance drill-down.")
