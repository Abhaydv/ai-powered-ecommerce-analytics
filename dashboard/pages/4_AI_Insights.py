import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Insights & Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Insights & Predictions")

st.markdown(
    "Explore AI-powered repeat purchase predictions "
    "and customer prioritization."
)

st.markdown("---")

# LOAD AI PREDICTIONS

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "repeat_purchase_test_predictions_v2.csv"
)

@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTION_PATH)


try:
    predictions_df = load_predictions()

except FileNotFoundError:
    st.error("AI predictions dataset not found.")
    st.stop()

    # CUSTOMER PRIORITY DISTRIBUTION

st.subheader("🎯 Customer Priority Distribution")

priority_counts = (
    predictions_df["priority"]
    .value_counts()
    .reset_index()
)

priority_counts.columns = ["Priority", "Customers"]

col1, col2, col3 = st.columns(3)

priority_map = dict(
    zip(priority_counts["Priority"], priority_counts["Customers"])
)

col1.metric("🔴 High Priority", f"{priority_map.get('High', 0):,}")
col2.metric("🟡 Medium Priority", f"{priority_map.get('Medium', 0):,}")
col3.metric("🟢 Low Priority", f"{priority_map.get('Low', 0):,}")

fig_priority = px.pie(
    priority_counts,
    names="Priority",
    values="Customers",
    hole=0.45,
    title="Customer Priority Distribution"
)

st.plotly_chart(fig_priority, width="stretch")

st.markdown("---")

# AI PREDICTION OVERVIEW

st.subheader("📊 AI Prediction Overview")

total_customers = len(predictions_df)

predicted_repeat = (
    predictions_df["predicted_repeat_purchase"].sum()
)

actual_repeat = (
    predictions_df["actual_repeat_purchase"].sum()
)

prediction_rate = (
    predicted_repeat / total_customers * 100
)

actual_rate = (
    actual_repeat / total_customers * 100
)

c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", f"{total_customers:,}")

c2.metric("Predicted Repeat Customers", f"{predicted_repeat:,}")

c3.metric("Actual Repeat Customers", f"{actual_repeat:,}")

st.write(
    f"**Predicted Repeat Purchase Rate:** {prediction_rate:.2f}%"
)

st.write(
    f"**Actual Repeat Purchase Rate:** {actual_rate:.2f}%"
)

st.markdown("---")


# MODEL PERFORMANCE METRICS

st.subheader("📈 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "96.45%")
col2.metric("Precision", "3.43%")
col3.metric("Recall", "12.73%")

col4, col5, col6 = st.columns(3)

col4.metric("F1 Score", "5.41%")
col5.metric("ROC-AUC", "61.58%")
col6.metric("PR-AUC", "2.73%")

st.info(
    "Note: The dataset is highly imbalanced. "
    "Therefore, accuracy alone is not a reliable measure "
    "of model performance. Precision, Recall and PR-AUC "
    "should also be considered."
)

st.markdown("---")

# CUSTOMER LEVEL PREDICTIONS

st.subheader("👥 Customer-Level AI Predictions")

priority_filter = st.selectbox(
    "Filter by Customer Priority",
    ["All", "High", "Medium", "Low"]
)

display_predictions = predictions_df.copy()

if priority_filter != "All":
    display_predictions = display_predictions[
        display_predictions["priority"] == priority_filter
    ]

display_predictions = display_predictions.sort_values(
    "repeat_purchase_probability",
    ascending=False
)

st.dataframe(
    display_predictions,
    width="stretch",
    hide_index=True
)

st.caption(
    "Predictions are based on a historical test cohort, "
    "not live customer data."
)

st.success("AI Insights & Predictions page loaded successfully!")