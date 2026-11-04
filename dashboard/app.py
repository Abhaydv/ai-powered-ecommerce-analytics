import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
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

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_ecommerce_dataset.csv"
)

PREDICTION_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "repeat_purchase_test_predictions_v2.csv"
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    master_df = pd.read_csv(
        MASTER_DATA_PATH,
        parse_dates=["order_purchase_timestamp"]
    )

    predictions_df = pd.read_csv(
        PREDICTION_DATA_PATH
    )

    return master_df, predictions_df


try:

    df, predictions = load_data()

except FileNotFoundError as error:

    st.error(f"Required dataset not found: {error}")
    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🛒 E-Commerce Analytics")

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-Powered E-Commerce Analytics Platform"
)

st.sidebar.markdown("### 📊 Dashboard Navigation")

st.sidebar.markdown("""
Explore different analytics modules:

- Sales Analytics
- Customer Analytics
- Product Analytics
- AI Insights & Predictions
""")

st.sidebar.markdown("---")

st.sidebar.subheader("🔎 Data Filters")

states = sorted(
    df["customer_state"].dropna().unique()
)

selected_states = st.sidebar.multiselect(
    "Select Customer State",
    states,
    default=[],
    placeholder="All States"
)

st.sidebar.caption(
    "Leave empty to view all states."
)


# =====================================================
# APPLY FILTERS
# =====================================================

if selected_states:

    filtered_df = df[
        df["customer_state"].isin(selected_states)
    ].copy()

else:

    filtered_df = df.copy()


# =====================================================
# HEADER
# =====================================================

st.title("🛒 E-Commerce Analytics Dashboard")

st.markdown(
    """
    Monitor business performance, understand customer behavior
    and explore AI-powered repeat purchase predictions.
    """
)

st.caption(
    "Business Intelligence | Sales Analytics | "
    "Customer Insights | Artificial Intelligence"
)

st.markdown("---")


# =====================================================
# ORDER LEVEL DATA
# =====================================================

order_df = (
    filtered_df.groupby("order_id")
    .agg(
        order_revenue=("order_revenue", "sum"),
        customer_unique_id=("customer_unique_id", "first"),
        order_status=("order_status", "first"),
        order_purchase_timestamp=(
            "order_purchase_timestamp", "first"
        )
    )
    .reset_index()
)


# =====================================================
# KPI CALCULATIONS
# =====================================================

total_orders = order_df["order_id"].nunique()

total_customers = (
    order_df["customer_unique_id"].nunique()
)

total_revenue = order_df["order_revenue"].sum()

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)


# =====================================================
# BUSINESS OVERVIEW
# =====================================================

st.subheader("📌 Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Orders",
    f"{total_orders:,}"
)

col2.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col3.metric(
    "Total Revenue",
    f"R$ {total_revenue / 1_000_000:.2f}M"
)

col4.metric(
    "Average Order Value",
    f"R$ {average_order_value:,.0f}"
)

st.markdown("---")


# =====================================================
# MONTHLY REVENUE TREND
# =====================================================

st.subheader("📈 Monthly Revenue Trend")

monthly_revenue = order_df.copy()

monthly_revenue["month"] = (
    monthly_revenue["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_revenue = (
    monthly_revenue.groupby("month")["order_revenue"]
    .sum()
    .reset_index()
)

# Exclude incomplete months (September–October 2018)

monthly_revenue = monthly_revenue[
    monthly_revenue["month"] <= "2018-08"
].copy()


# =====================================================
# MONTHLY REVENUE CHART
# =====================================================

if not monthly_revenue.empty:

    fig = px.line(
        monthly_revenue,
        x="month",
        y="order_revenue",
        markers=True,
        title="Monthly Revenue Performance",
        labels={
            "month": "Month",
            "order_revenue": "Revenue (R$)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Revenue (R$)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("No monthly revenue data available.")

st.markdown("---")


# =====================================================
# MONTHLY ORDERS TREND
# =====================================================

st.subheader("📦 Monthly Orders Trend")

monthly_orders = order_df.copy()

monthly_orders["month"] = (
    monthly_orders["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_orders = (
    monthly_orders.groupby("month")["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

# Exclude incomplete months
monthly_orders = monthly_orders[
    monthly_orders["month"] <= "2018-08"
].copy()

if not monthly_orders.empty:

    fig = px.bar(
        monthly_orders,
        x="month",
        y="total_orders",
        title="Monthly Order Volume",
        labels={
            "month": "Month",
            "total_orders": "Total Orders"
        },
        text="total_orders"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Number of Orders",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("No monthly order data available.")

st.markdown("---")

# =====================================================
# STATE-WISE REVENUE ANALYSIS
# =====================================================

st.markdown("---")

st.subheader("🌎 State-wise Revenue Analysis")

state_revenue = (
    filtered_df.groupby("customer_state")
    .agg(
        total_revenue=("order_revenue", "sum"),
        total_orders=("order_id", "nunique"),
        total_customers=("customer_unique_id", "nunique")
    )
    .reset_index()
)

state_revenue = state_revenue.sort_values(
    "total_revenue",
    ascending=False
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        state_revenue.head(10),
        x="customer_state",
        y="total_revenue",
        title="Top 10 States by Revenue",
        labels={
            "customer_state": "State",
            "total_revenue": "Revenue (R$)"
        },
        text_auto=".2s"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Customer State",
        yaxis_title="Revenue (R$)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.bar(
        state_revenue.head(10),
        x="customer_state",
        y="total_orders",
        title="Top 10 States by Orders",
        labels={
            "customer_state": "State",
            "total_orders": "Total Orders"
        },
        text_auto=True
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Customer State",
        yaxis_title="Number of Orders"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")


# =====================================================
# STATE PERFORMANCE TABLE
# =====================================================

st.subheader("📋 State Performance Details")

display_state = state_revenue.copy()

display_state["total_revenue"] = (
    display_state["total_revenue"].round(2)
)

display_state = display_state.rename(
    columns={
        "customer_state": "State",
        "total_revenue": "Total Revenue (R$)",
        "total_orders": "Total Orders",
        "total_customers": "Total Customers"
    }
)

st.dataframe(
    display_state,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.markdown("---")

st.subheader("💡 Key Business Insights")

if not state_revenue.empty and total_orders > 0:

    top_revenue_state = state_revenue.iloc[0]

    top_order_state = state_revenue.sort_values(
        "total_orders",
        ascending=False
    ).iloc[0]

    top_state_share = (
        top_revenue_state["total_revenue"]
        / total_revenue * 100
        if total_revenue > 0
        else 0
    )

    repeat_customer_count = (
        predictions["actual_repeat_purchase"].sum()
    )

    actual_repeat_rate = (
        repeat_customer_count
        / len(predictions) * 100
        if len(predictions) > 0
        else 0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            **🏆 Revenue Leader**

            {top_revenue_state['customer_state']} is the
            highest revenue-generating state with
            R$ {top_revenue_state['total_revenue']:,.2f}.

            It contributes {top_state_share:.2f}% of
            the selected dataset's revenue.
            """
        )

    with col2:
        st.info(
            f"""
            **📦 Order Volume Leader**

            {top_order_state['customer_state']} leads in
            order volume with
            {top_order_state['total_orders']:,} orders.
            """
        )

    st.success(
        f"""
        **🤖 AI Model Observation**

        The historical test cohort has an actual repeat
        purchase rate of {actual_repeat_rate:.2f}%.
        Customer retention is an important opportunity
        for future business growth.
        """
    )

else:
    st.info("Insights will appear when data is available.")
