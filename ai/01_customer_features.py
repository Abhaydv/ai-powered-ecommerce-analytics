# ============================================================
# CUSTOMER FEATURE ENGINEERING
# AI-POWERED E-COMMERCE ANALYTICS
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_ecommerce_dataset.csv"
)

ORDER_ITEMS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "order_items_clean.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_features.csv"
)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

print("=" * 70)
print("CUSTOMER FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading master dataset...")

master_df = pd.read_csv(MASTER_PATH)

master_df["order_purchase_timestamp"] = pd.to_datetime(
    master_df["order_purchase_timestamp"],
    errors="coerce"
)

print("Master rows:", len(master_df))
print("Master columns:", len(master_df.columns))


# ============================================================
# LOAD ORDER ITEMS
# ============================================================

print("\nLoading order items...")

order_items = pd.read_csv(
    ORDER_ITEMS_PATH
)

print("Order item rows:", len(order_items))


# ============================================================
# CALCULATE ITEMS PER ORDER
# ============================================================

item_counts = (
    order_items
    .groupby("order_id")
    .size()
    .rename("order_item_count")
    .reset_index()
)


# ============================================================
# CREATE ORDER-LEVEL DATA
# ============================================================

required_columns = [
    "order_id",
    "customer_unique_id",
    "order_purchase_timestamp",
    "order_revenue",
    "review_score",
    "delivery_days"
]

missing_columns = [
    col
    for col in required_columns
    if col not in master_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


orders = (
    master_df[required_columns]
    .drop_duplicates("order_id")
    .copy()
)


# ============================================================
# MERGE ITEM COUNTS
# ============================================================

orders = orders.merge(
    item_counts,
    on="order_id",
    how="left"
)

orders["order_item_count"] = (
    orders["order_item_count"]
    .fillna(0)
)


# ============================================================
# CUSTOMER-LEVEL FEATURES
# ============================================================

customer_features = (
    orders
    .groupby("customer_unique_id")
    .agg(
        Frequency=("order_id", "nunique"),

        Monetary=("order_revenue", "sum"),

        Avg_Order_Value=("order_revenue", "mean"),

        Avg_Review_Score=("review_score", "mean"),

        Avg_Delivery_Days=("delivery_days", "mean"),

        Total_Items=("order_item_count", "sum"),

        Last_Purchase=(
            "order_purchase_timestamp",
            "max"
        ),

        First_Purchase=(
            "order_purchase_timestamp",
            "min"
        )
    )
    .reset_index()
)


# ============================================================
# RECENCY
# ============================================================

cutoff_date = orders[
    "order_purchase_timestamp"
].max()

customer_features["Recency"] = (
    cutoff_date
    - customer_features["Last_Purchase"]
).dt.days


# ============================================================
# CUSTOMER LIFETIME
# ============================================================

customer_features["Customer_Lifetime_Days"] = (
    customer_features["Last_Purchase"]
    - customer_features["First_Purchase"]
).dt.days


# ============================================================
# ITEMS PER ORDER
# ============================================================

customer_features["Items_Per_Order"] = (
    customer_features["Total_Items"]
    / customer_features["Frequency"]
)


# ============================================================
# REPEAT PURCHASE
# ============================================================

customer_features["repeat_purchase"] = (
    customer_features["Frequency"] > 1
).astype(int)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

customer_features = customer_features[
    [
        "customer_unique_id",
        "Frequency",
        "Monetary",
        "Avg_Order_Value",
        "Avg_Review_Score",
        "Avg_Delivery_Days",
        "Total_Items",
        "Last_Purchase",
        "First_Purchase",
        "Recency",
        "Customer_Lifetime_Days",
        "Items_Per_Order",
        "repeat_purchase"
    ]
]


# ============================================================
# SAVE
# ============================================================

customer_features.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER FEATURES CREATED")
print("=" * 70)

print("Rows:", len(customer_features))
print("Columns:", len(customer_features.columns))

print("\nFeatures:")
print(customer_features.columns.tolist())

print("\nTarget distribution:")
print(
    customer_features["repeat_purchase"]
    .value_counts()
)

print("\nMissing values:")
print(
    customer_features.isnull().sum()
)

print("\nSaved to:")
print(OUTPUT_PATH)