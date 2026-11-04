import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# PAGE CONFIGURATION

st.set_page_config(
page_title="Product Analytics",
page_icon="📦",
layout="wide"
)

st.title("📦 Product Analytics")
st.markdown(
"Analyze product categories, revenue contribution and sales performance."
)
st.markdown("---")

# LOAD DATA

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "master_ecommerce_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()

except FileNotFoundError:
    st.error("Master dataset not found.")
    st.stop()

# FILTER INVALID CATEGORIES

df = df.dropna(subset=["product_category_name_english"]).copy()

# SIDEBAR FILTER

st.sidebar.header("Product Filters")

categories = sorted(
df["product_category_name_english"].unique()
)

selected_categories = st.sidebar.multiselect(
"Select Categories",
categories,
placeholder="All categories"
)

if selected_categories:
    df = df[
        df["product_category_name_english"].isin(
            selected_categories
        )
    ].copy()

# CATEGORY LEVEL ANALYSIS

category_df = (
df.groupby("product_category_name_english")
.agg(
revenue=("order_revenue", "sum"),
orders=("order_id", "nunique"),
products=("product_id", "nunique"),
average_price=("price", "mean")
)
.reset_index()
)

category_df = category_df.sort_values(
"revenue", ascending=False
)

# KPIs

total_revenue = category_df["revenue"].sum()
total_orders = df["order_id"].nunique()
total_products = df["product_id"].nunique()
total_categories = df["product_category_name_english"].nunique()

st.subheader("Product Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"R$ {total_revenue / 1_000_000:.2f}M")
c2.metric("Total Orders", f"{total_orders:,}")
c3.metric("Unique Products", f"{total_products:,}")
c4.metric("Categories", f"{total_categories:,}")

st.markdown("---")

# TOP CATEGORIES BY REVENUE

st.subheader("🏆 Top 10 Categories by Revenue")

top_categories = category_df.head(10).sort_values("revenue")

fig_revenue = px.bar(
top_categories,
x="revenue",
y="product_category_name_english",
orientation="h",
title="Top Categories by Revenue",
labels={
"revenue": "Revenue (BRL)",
"product_category_name_english": "Category"
}
)

st.plotly_chart(fig_revenue, width="stretch")

# TOP CATEGORIES BY ORDERS

st.subheader("📊 Top 10 Categories by Orders")

top_orders = category_df.nlargest(10, "orders").sort_values("orders")

fig_orders = px.bar(
top_orders,
x="orders",
y="product_category_name_english",
orientation="h",
title="Top Categories by Order Volume",
labels={
"orders": "Number of Orders",
"product_category_name_english": "Category"
}
)

st.plotly_chart(fig_orders, width="stretch")

# CATEGORY PRICE ANALYSIS

st.subheader("💰 Average Product Price by Category")

price_df = category_df.nlargest(10, "revenue")

fig_price = px.bar(
price_df.sort_values("average_price"),
x="average_price",
y="product_category_name_english",
orientation="h",
title="Average Price of Top Revenue Categories",
labels={
"average_price": "Average Price (BRL)",
"product_category_name_english": "Category"
}
)

st.plotly_chart(fig_price, width="stretch")

# CATEGORY DATA TABLE

st.markdown("---")
st.subheader("Category Performance Details")

display_df = category_df.copy()

display_df["revenue"] = display_df["revenue"].round(2)
display_df["average_price"] = display_df["average_price"].round(2)

st.dataframe(
display_df,
width="stretch",
hide_index=True
)

st.success("Product Analytics page loaded successfully!")

st.caption(
"Revenue is aggregated at order-item level. "
"Category order counts use unique order IDs."
)
