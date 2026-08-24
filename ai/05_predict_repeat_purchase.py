# REPEAT PURCHASE PREDICTION

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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
    / "repeat_purchase_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CUTOFF_DATE = pd.Timestamp("2018-03-01")

SELECTED_THRESHOLD = 0.04

FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Avg_Order_Value",
    "Avg_Review_Score",
    "Avg_Delivery_Days",
    "Total_Items",
    "Customer_Lifetime_Days",
    "Items_Per_Order"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("REPEAT PURCHASE PREDICTION")
print("=" * 70)

print("\nLoading master dataset...")

master_df = pd.read_csv(
    MASTER_PATH
)

master_df["order_purchase_timestamp"] = pd.to_datetime(
    master_df["order_purchase_timestamp"],
    errors="coerce"
)

print(
    "Master rows:",
    len(master_df)
)


# ============================================================
# LOAD ORDER ITEMS
# ============================================================

print("\nLoading order items...")

order_items = pd.read_csv(
    ORDER_ITEMS_PATH
)

print(
    "Order item rows:",
    len(order_items)
)


# ============================================================
# ITEM COUNT PER ORDER
# ============================================================

item_counts = (
    order_items
    .groupby("order_id")
    .size()
    .rename("Total_Items")
    .reset_index()
)


# ============================================================
# ORDER DATA
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
    master_df[
        required_columns
    ]
    .drop_duplicates(
        "order_id"
    )
    .copy()
)


orders = orders.merge(
    item_counts,
    on="order_id",
    how="left"
)


orders["Total_Items"] = (
    orders["Total_Items"]
    .fillna(0)
)


# ============================================================
# PAST ORDERS
# ============================================================

past_orders = orders[
    orders["order_purchase_timestamp"]
    < CUTOFF_DATE
].copy()


print("\nPast Orders:", len(past_orders))


# ============================================================
# CUSTOMER FEATURES
# ============================================================

customer_features = (
    past_orders
    .groupby(
        "customer_unique_id"
    )
    .agg(
        Frequency=(
            "order_id",
            "nunique"
        ),

        Monetary=(
            "order_revenue",
            "sum"
        ),

        Avg_Order_Value=(
            "order_revenue",
            "mean"
        ),

        Avg_Review_Score=(
            "review_score",
            "mean"
        ),

        Avg_Delivery_Days=(
            "delivery_days",
            "mean"
        ),

        Total_Items=(
            "Total_Items",
            "sum"
        ),

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

customer_features["Recency"] = (
    CUTOFF_DATE
    - customer_features["Last_Purchase"]
).dt.days


# ============================================================
# CUSTOMER LIFETIME
# ============================================================

customer_features[
    "Customer_Lifetime_Days"
] = (
    customer_features["Last_Purchase"]
    - customer_features["First_Purchase"]
).dt.days


# ============================================================
# ITEMS PER ORDER
# ============================================================

customer_features[
    "Items_Per_Order"
] = (
    customer_features["Total_Items"]
    / customer_features["Frequency"]
)


# ============================================================
# MODEL DATA
# ============================================================

X = customer_features[
    FEATURE_COLS
].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

delivery_median = (
    X["Avg_Delivery_Days"]
    .median()
)

X["Avg_Delivery_Days"] = (
    X["Avg_Delivery_Days"]
    .fillna(
        delivery_median
    )
)


# ============================================================
# TRAIN MODEL
# ============================================================

# IMPORTANT:
# This is only used to recreate the same model
# that was evaluated during threshold optimization.

# We need a target for training. Since the final prediction
# model needs labels, use customers' historical repeat behavior
# within the past data as a training signal.

past_customer_orders = (
    past_orders
    .groupby(
        "customer_unique_id"
    )["order_id"]
    .nunique()
)


training_target = (
    past_customer_orders
    .reindex(
        customer_features[
            "customer_unique_id"
        ]
    )
    .fillna(0)
    .gt(1)
    .astype(int)
    .values
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        training_target,
        test_size=0.20,
        stratify=training_target,
        random_state=42
    )
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

lr_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


lr_model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

customer_features[
    "repeat_purchase_probability"
] = (
    lr_model
    .predict_proba(X)[:, 1]
)


# ============================================================
# APPLY SELECTED THRESHOLD
# ============================================================

customer_features[
    "predicted_repeat_purchase"
] = (
    customer_features[
        "repeat_purchase_probability"
    ]
    >= SELECTED_THRESHOLD
).astype(int)


# ============================================================
# PRIORITY
# ============================================================

def assign_priority(probability):

    if probability >= 0.10:
        return "High"

    elif probability >= SELECTED_THRESHOLD:
        return "Medium"

    else:
        return "Low"


customer_features[
    "priority"
] = (
    customer_features[
        "repeat_purchase_probability"
    ]
    .apply(assign_priority)
)


# ============================================================
# FINAL OUTPUT
# ============================================================

prediction_df = customer_features[
    [
        "customer_unique_id",
        "Frequency",
        "Monetary",
        "Avg_Order_Value",
        "Avg_Review_Score",
        "Avg_Delivery_Days",
        "Total_Items",
        "Recency",
        "Customer_Lifetime_Days",
        "Items_Per_Order",
        "repeat_purchase_probability",
        "predicted_repeat_purchase",
        "priority"
    ]
].copy()


# ============================================================
# SORT BY PROBABILITY
# ============================================================

prediction_df = (
    prediction_df
    .sort_values(
        "repeat_purchase_probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE
# ============================================================

prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "Selected Threshold:",
    SELECTED_THRESHOLD
)

print(
    "Total Customers:",
    len(prediction_df)
)

print(
    "Predicted Repeat Customers:",
    prediction_df[
        "predicted_repeat_purchase"
    ].sum()
)

print("\nPriority Distribution:")

print(
    prediction_df[
        "priority"
    ].value_counts()
)


print("\nTop 10 Customers:")

print(
    prediction_df
    .head(10)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(
    OUTPUT_PATH
)