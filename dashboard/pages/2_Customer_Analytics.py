import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="👥",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.04);
    min-height: 120px;
}

[data-testid="stMetricLabel"] {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.3rem, 2vw, 2rem);
    color: #0f172a;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}

h1, h2, h3 {
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_ecommerce_dataset.csv"
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH,
        parse_dates=["order_purchase_timestamp"]
    )


try:

    df = load_data()

except FileNotFoundError:

    st.error("Master dataset not found.")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.title("👥 Customer Analytics")

st.markdown(
    "Understand customer purchasing behavior, "
    "loyalty, purchase frequency and customer value."
)

st.caption(
    "Customer Segmentation | Loyalty Analysis | Revenue Insights"
)

st.markdown("---")


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("👥 Customer Analytics")

st.sidebar.markdown("---")

st.sidebar.subheader("🔎 Customer Filters")

states = sorted(
    df["customer_state"].dropna().unique()
)

selected_states = st.sidebar.multiselect(
    "Select Customer States",
    states,
    default=[],
    placeholder="All States"
)

if selected_states:

    filtered_df = df[
        df["customer_state"].isin(selected_states)
    ].copy()

else:

    filtered_df = df.copy()

st.sidebar.caption(
    "Leave empty to view all states."
)


# =====================================================
# ORDER LEVEL DATA
# =====================================================

order_df = (
    filtered_df.groupby("order_id")
    .agg(
        customer_id=("customer_unique_id", "first"),
        revenue=("order_revenue", "sum"),
        order_date=("order_purchase_timestamp", "first"),
        state=("customer_state", "first")
    )
    .reset_index()
)


# =====================================================
# CUSTOMER LEVEL DATA
# =====================================================

customer_df = (
    order_df.groupby("customer_id")
    .agg(
        total_orders=("order_id", "nunique"),
        total_revenue=("revenue", "sum"),
        average_order_value=("revenue", "mean"),
        last_purchase=("order_date", "max")
    )
    .reset_index()
)


customer_df["customer_segment"] = customer_df[
    "total_orders"
].apply(
    lambda x: (
        "Repeat Customer"
        if x > 1
        else "One-Time Customer"
    )
)


# =====================================================
# CUSTOMER KPIs
# =====================================================

total_customers = customer_df["customer_id"].nunique()

repeat_customers = (
    customer_df["total_orders"] > 1
).sum()

one_time_customers = (
    customer_df["total_orders"] == 1
).sum()

repeat_rate = (
    repeat_customers / total_customers * 100
    if total_customers > 0
    else 0
)

average_customer_revenue = (
    customer_df["total_revenue"].mean()
    if total_customers > 0
    else 0
)


# =====================================================
# CUSTOMER OVERVIEW
# =====================================================

st.subheader("📌 Customer Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

c2.metric(
    "Repeat Customers",
    f"{repeat_customers:,}"
)

c3.metric(
    "One-Time Customers",
    f"{one_time_customers:,}"
)

c4.metric(
    "Repeat Customer Rate",
    f"{repeat_rate:.2f}%"
)

st.markdown("---")


# =====================================================
# CUSTOMER REVENUE METRICS
# =====================================================

st.subheader("💰 Customer Revenue Insights")

r1, r2 = st.columns(2)

r1.metric(
    "Average Revenue per Customer",
    f"R$ {average_customer_revenue:,.2f}"
)

r2.metric(
    "Total Customer Revenue",
    f"R$ {customer_df['total_revenue'].sum():,.2f}"
)

st.markdown("---")


# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

st.subheader("📊 Customer Segmentation")

col1, col2 = st.columns(2)

with col1:

    segment_df = (
        customer_df["customer_segment"]
        .value_counts()
        .reset_index()
    )

    segment_df.columns = ["Segment", "Customers"]

    fig = px.pie(
        segment_df,
        names="Segment",
        values="Customers",
        hole=0.55,
        title="One-Time vs Repeat Customers"
    )

    st.plotly_chart(fig, use_container_width=True)


with col2:

    fig = px.histogram(
        customer_df,
        x="total_revenue",
        nbins=40,
        title="Customer Revenue Distribution",
        labels={
            "total_revenue": "Customer Revenue",
            "count": "Number of Customers"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# =====================================================
# PURCHASE FREQUENCY
# =====================================================

st.subheader("🛒 Customer Purchase Frequency")

frequency_df = (
    customer_df["total_orders"]
    .value_counts()
    .reset_index()
)

frequency_df.columns = [
    "Number of Orders",
    "Number of Customers"
]

frequency_df = frequency_df.sort_values(
    "Number of Orders"
)

fig = px.bar(
    frequency_df,
    x="Number of Orders",
    y="Number of Customers",
    title="Customers by Number of Orders",
    text="Number of Customers"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# =====================================================
# TOP CUSTOMERS
# =====================================================

st.subheader("🏆 Top 10 Customers by Revenue")

top_customers = customer_df.sort_values(
    "total_revenue",
    ascending=False
).head(10).copy()

top_customers = top_customers.rename(
    columns={
        "customer_id": "Customer ID",
        "total_orders": "Total Orders",
        "total_revenue": "Total Revenue",
        "average_order_value": "Average Order Value",
        "last_purchase": "Last Purchase",
        "customer_segment": "Customer Segment"
    }
)

st.dataframe(
    top_customers,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.markdown("---")

st.subheader("💡 Business Insights")

if total_customers > 0:

    st.info(
        f"""
        - **{repeat_rate:.2f}%** of customers made more than one purchase.
        - **{one_time_customers:,}** customers purchased only once.
        - Average revenue per customer is **R$ {average_customer_revenue:,.2f}**.
        - There is an opportunity to improve customer retention through
          loyalty programs, personalized offers and targeted campaigns.
        """
    )