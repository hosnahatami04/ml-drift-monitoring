"""Streamlit dashboard over the live store and the evaluation report.

Run with: streamlit run src/driftmon/dashboard/app.py
"""

from pathlib import Path

import streamlit as st

from driftmon.config import load_settings
from driftmon.dashboard import data

REFRESH_SECONDS = 5

settings = load_settings()
db_path = settings.db_path
reports_dir = settings.reports_dir

st.set_page_config(page_title="driftmon", layout="wide")
st.title("driftmon")
st.caption(f"Store: `{db_path}` · reports: `{reports_dir}`")

live_tab, evaluation_tab = st.tabs(["Live monitoring", "Evaluation results"])


@st.fragment(run_every=REFRESH_SECONDS)
def live_view() -> None:
    status = data.load_status(db_path)
    columns = st.columns(4)
    columns[0].metric("Predictions", f"{status.predictions:,}")
    columns[1].metric("Windows processed", status.processed_windows)
    columns[2].metric("Last window", "-" if status.last_window is None else status.last_window)
    columns[3].metric("Alerts", status.alerts)
    if status.latest_alert is not None:
        alert = status.latest_alert
        st.warning(
            f"Latest alert: {alert['type']} ({alert['severity']}) on window "
            f"{alert['window_id']} {alert['features']}"
        )

    statistics = data.load_feature_statistics(db_path)
    st.subheader("Detector statistics per window")
    if statistics.empty:
        st.info("No processed windows yet.")
    else:
        methods = sorted(statistics["method"].unique())
        method = st.selectbox("Test", methods, index=methods.index("ks") if "ks" in methods else 0)
        chosen = statistics[statistics["method"] == method]
        st.line_chart(chosen.pivot_table(index="window_id", columns="feature", values="statistic"))
        flagged = chosen[chosen["flagged"]]
        if not flagged.empty:
            st.caption("Flagged in these windows:")
            st.dataframe(flagged, hide_index=True, width="stretch")

    st.subheader("Alert timeline")
    alerts = data.load_alerts(db_path)
    if alerts.empty:
        st.info("No alerts.")
    else:
        st.scatter_chart(alerts, x="window_id", y="type", color="severity")
        st.dataframe(alerts, hide_index=True, width="stretch")

    st.subheader("Model performance (windows with all labels)")
    performance = data.load_performance(db_path)
    if performance.empty:
        st.info("No fully labelled windows yet.")
    else:
        left, right = st.columns(2)
        left.caption("AUC")
        left.line_chart(performance, x="window_id", y="auc")
        right.caption("Error rate")
        right.line_chart(performance, x="window_id", y="error_rate")


def evaluation_view() -> None:
    document = data.load_evaluation(reports_dir / "results.json")
    if document is None:
        st.info("No evaluation results found. Run `make evaluate` first.")
        return
    p = document.provenance
    st.caption(
        f"Evaluation seeds {p['seeds'][0]}-{p['seeds'][-1]}, model `{p['artifact']}`, "
        f"reference AUC {p['reference_auc']:.4f}"
    )
    st.subheader("KPIs")
    st.dataframe(data.kpi_table(document), hide_index=True, width="stretch")
    st.subheader("By scenario and severity")
    st.dataframe(data.cell_table(document), hide_index=True, width="stretch")
    figures = sorted(Path(reports_dir / "figures").glob("*.png"))
    for figure in figures:
        st.image(str(figure), caption=figure.stem.replace("_", " "))


with live_tab:
    live_view()

with evaluation_tab:
    evaluation_view()
