import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📈",
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

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["order_purchase_timestamp"]
    )

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error("Master dataset not found. Please check the data path.")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.title("📈 Sales Analytics")

st.markdown(
    "Analyze revenue trends, order performance, "
    "sales growth and product category contribution."
)

st.caption(
    "Sales Performance | Revenue Analysis | Order Trends"
)

st.markdown("---")


# =====================================================
# ORDER LEVEL DATA
# =====================================================

order_df = (
    df.groupby("order_id")
    .agg(
        revenue=("order_revenue", "sum"),
        customer_id=("customer_unique_id", "first"),
        order_date=("order_purchase_timestamp", "first"),
        state=("customer_state", "first")
    )
    .reset_index()
)


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("📈 Sales Analytics")

st.sidebar.markdown("---")

st.sidebar.subheader("🔎 Sales Filters")

states = sorted(
    order_df["state"].dropna().unique()
)

selected_states = st.sidebar.multiselect(
    "Select States",
    states,
    default=[],
    placeholder="All States"
)

# State filter

if selected_states:

    filtered_orders = order_df[
        order_df["state"].isin(selected_states)
    ].copy()

    filtered_df = df[
        df["customer_state"].isin(selected_states)
    ].copy()

else:

    filtered_orders = order_df.copy()
    filtered_df = df.copy()


# Date range

min_date = filtered_orders["order_date"].min().date()

actual_max_date = filtered_orders["order_date"].max().date()

# Default end date: August 2018

complete_month_end = pd.Timestamp(
    "2018-08-31"
).date()

default_max_date = min(
    actual_max_date,
    complete_month_end
)

if default_max_date < min_date:
    default_max_date = actual_max_date

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, default_max_date),
    min_value=min_date,
    max_value=actual_max_date
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:

    st.warning("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:

    st.error("Start date cannot be after end date.")
    st.stop()


# Apply date filters

filtered_orders = filtered_orders[
    filtered_orders["order_date"].dt.date.between(
        start_date,
        end_date
    )
].copy()

filtered_df = filtered_df[
    filtered_df["order_purchase_timestamp"].dt.date.between(
        start_date,
        end_date
    )
].copy()


st.sidebar.markdown("---")

st.sidebar.caption(
    "Filters apply to all sales KPIs and charts."
)


# =====================================================
# SALES KPIs
# =====================================================

total_revenue = filtered_orders["revenue"].sum()

total_orders = filtered_orders["order_id"].nunique()

total_customers = filtered_orders["customer_id"].nunique()

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)


# =====================================================
# SALES PERFORMANCE OVERVIEW
# =====================================================

st.subheader("📌 Sales Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Revenue",
    f"R$ {total_revenue / 1_000_000:.2f}M"
)

c2.metric(
    "Total Orders",
    f"{total_orders:,}"
)

c3.metric(
    "Total Customers",
    f"{total_customers:,}"
)

c4.metric(
    "Average Order Value",
    f"R$ {average_order_value:,.0f}"
)

st.markdown("---")


# =====================================================
# MONTHLY REVENUE TREND
# =====================================================

st.subheader("📈 Monthly Revenue Trend")

monthly_sales = filtered_orders.copy()

monthly_sales["month"] = (
    monthly_sales["order_date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_revenue = (
    monthly_sales.groupby("month")["revenue"]
    .sum()
    .reset_index()
)

# Exclude incomplete months

monthly_revenue = monthly_revenue[
    monthly_revenue["month"] <= "2018-08"
].copy()


fig_revenue = px.line(
    monthly_revenue,
    x="month",
    y="revenue",
    markers=True,
    title="Monthly Revenue Performance",
    template="plotly_white"
)

fig_revenue.update_traces(
    line=dict(width=3),
    marker=dict(size=8)
)

fig_revenue.update_layout(
    xaxis_title="Month",
    yaxis_title="Revenue (BRL)",
    hovermode="x unified",
    title_x=0.02,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(
    fig_revenue,
    width="stretch"
)

st.caption(
    "September and October 2018 are excluded "
    "because they contain incomplete monthly data."
)

st.markdown("---")


# =====================================================
# MONTHLY ORDER VOLUME
# =====================================================

st.subheader("📦 Monthly Order Volume")

monthly_orders = (
    monthly_sales.groupby("month")["order_id"]
    .nunique()
    .reset_index(name="orders")
)

monthly_orders = monthly_orders[
    monthly_orders["month"] <= "2018-08"
].copy()


fig_orders = px.bar(
    monthly_orders,
    x="month",
    y="orders",
    title="Monthly Order Performance",
    text_auto=True,
    template="plotly_white"
)

fig_orders.update_layout(
    xaxis_title="Month",
    yaxis_title="Number of Orders",
    title_x=0.02,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(
    fig_orders,
    width="stretch"
)

st.markdown("---")


# =====================================================
# TOP 10 PRODUCT CATEGORIES
# =====================================================

st.subheader("🏆 Top 10 Product Categories by Revenue")

category_df = filtered_df.dropna(
    subset=["product_category_name_english"]
).copy()


category_sales = (
    category_df.groupby("product_category_name_english")
    .agg(
        revenue=("order_revenue", "sum"),
        orders=("order_id", "nunique"),
        products=("product_id", "nunique")
    )
    .reset_index()
)

category_sales = category_sales.nlargest(
    10,
    "revenue"
)

fig_category = px.bar(
    category_sales.sort_values("revenue"),
    x="revenue",
    y="product_category_name_english",
    orientation="h",
    title="Top Categories by Revenue",
    labels={
        "revenue": "Revenue (BRL)",
        "product_category_name_english": "Product Category"
    },
    template="plotly_white"
)

fig_category.update_layout(
    title_x=0.02,
    xaxis_title="Revenue (BRL)",
    yaxis_title="Product Category",
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(
    fig_category,
    width="stretch"
)


# =====================================================
# CATEGORY PERFORMANCE TABLE
# =====================================================

st.markdown("---")

st.subheader("📋 Category Performance Details")

display_df = category_sales.copy()

display_df["revenue"] = display_df["revenue"].round(2)

display_df = display_df.sort_values(
    "revenue",
    ascending=False
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.markdown("---")

st.subheader("💡 Sales Business Insights")

if not category_sales.empty:

    top_category = category_sales.loc[
        category_sales["revenue"].idxmax()
    ]

    st.info(
        f"Top revenue-generating category: "
        f"{top_category['product_category_name_english']}. "
        f"It generated R$ {top_category['revenue']:,.2f} "
        f"in revenue during the selected period."
    )

else:

    st.info(
        "No product category data available "
        "for the selected filters."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:15px;
        font-size:14px;
    ">
        <b>Sales Analytics Module</b><br>
        Python | Pandas | Plotly | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

st.success(
    "Sales Analytics page loaded successfully!"
)