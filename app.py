"""Streamlit app for credit card fraud scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_COLUMNS,
    load_metadata,
    score_dataframe,
)


st.set_page_config(
    page_title="Fraud Transaction Scoring",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(DEFAULT_MODEL_PATH)
    scaler = joblib.load(DEFAULT_SCALER_PATH)
    metadata = load_metadata(DEFAULT_METADATA_PATH)
    return model, scaler, metadata


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("Credit Card Fraud Scoring")
st.caption("Upload transaction rows and score fraud probability with the trained model.")

artifact_missing = [
    path
    for path in [DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH]
    if not path.exists()
]

if artifact_missing:
    st.error(
        "Missing model artifacts: "
        + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in artifact_missing)
    )
    st.stop()

model, scaler, metadata = load_artifacts()
default_threshold = float(metadata.get("threshold", 0.5))
model_name = metadata.get("model_name", "trained model")

with st.sidebar:
    st.header("Scoring")
    threshold = st.slider(
        "Fraud threshold",
        min_value=0.01,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
    )
    st.metric("Model", model_name)
    st.metric("Default threshold", f"{default_threshold:.2f}")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is None:
    st.info(
        "CSV must include Time, Amount, and anonymized features V1 through V28."
    )
    st.dataframe(
        pd.DataFrame(columns=FEATURE_COLUMNS),
        use_container_width=True,
        hide_index=True,
    )
    st.stop()

try:
    input_df = pd.read_csv(uploaded_file)
    scored_df = score_dataframe(input_df, model, scaler, threshold)
except Exception as exc:
    st.error(str(exc))
    st.stop()

fraud_count = int(scored_df["fraud_prediction"].sum())
fraud_rate = fraud_count / len(scored_df) if len(scored_df) else 0
avg_probability = scored_df["fraud_probability"].mean()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows scored", f"{len(scored_df):,}")
metric_cols[1].metric("Flagged fraud", f"{fraud_count:,}")
metric_cols[2].metric("Flag rate", f"{fraud_rate:.2%}")
metric_cols[3].metric("Avg probability", f"{avg_probability:.3f}")

st.subheader("Highest Risk Transactions")
display_columns = [
    "fraud_probability",
    "fraud_prediction",
    "threshold",
] + [column for column in ["Time", "Amount", "Class"] if column in scored_df.columns]

top_risk_df = scored_df.sort_values("fraud_probability", ascending=False)
st.dataframe(
    top_risk_df.loc[:, display_columns].head(25),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Scored Data")
st.dataframe(scored_df.head(100), use_container_width=True, hide_index=True)

st.download_button(
    "Download scored CSV",
    data=to_csv_bytes(scored_df),
    file_name="scored_transactions.csv",
    mime="text/csv",
)
