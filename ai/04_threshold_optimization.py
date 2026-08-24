# THRESHOLD OPTIMIZATION

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


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
    / "threshold_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CUTOFF_DATE = pd.Timestamp("2018-03-01")

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
# LOAD MASTER DATA
# ============================================================

print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print("\nLoading master dataset...")

master_df = pd.read_csv(
    MASTER_PATH
)

master_df["order_purchase_timestamp"] = pd.to_datetime(
    master_df["order_purchase_timestamp"],
    errors="coerce"
)

print("Master rows:", len(master_df))


# ============================================================
# LOAD ORDER ITEMS
# ============================================================

print("\nLoading order items...")

order_items = pd.read_csv(
    ORDER_ITEMS_PATH
)

print("Order item rows:", len(order_items))


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
# CREATE ORDER-LEVEL DATASET
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


# ============================================================
# MERGE ITEM COUNTS
# ============================================================

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
# TIME-BASED SPLIT
# ============================================================

past_orders = orders[
    orders["order_purchase_timestamp"]
    < CUTOFF_DATE
].copy()

future_orders = orders[
    orders["order_purchase_timestamp"]
    >= CUTOFF_DATE
].copy()


print("\n" + "=" * 70)
print("TIME-BASED SPLIT")
print("=" * 70)

print(
    "Cutoff Date:",
    CUTOFF_DATE
)

print(
    "Past Orders:",
    len(past_orders)
)

print(
    "Future Orders:",
    len(future_orders)
)


# ============================================================
# FUTURE PURCHASING CUSTOMERS
# ============================================================

future_customers = set(
    future_orders[
        "customer_unique_id"
    ]
    .dropna()
    .unique()
)

print(
    "Future purchasing customers:",
    len(future_customers)
)


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
# TARGET
# ============================================================

customer_features[
    "repeat_purchase"
] = (
    customer_features[
        "customer_unique_id"
    ]
    .isin(
        future_customers
    )
    .astype(int)
)


# ============================================================
# CREATE X AND y
# ============================================================

X = customer_features[
    FEATURE_COLS
].copy()

y = customer_features[
    "repeat_purchase"
].copy()


# ============================================================
# DATASET DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD SCRIPT DATA CHECK")
print("=" * 70)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print("\nTarget distribution:")

print(
    y.value_counts()
)

print("\nTarget percentage:")

print(
    y.value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")

print(
    X.isna().sum()
)


# ============================================================
# IMPUTE DELIVERY
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


print(
    "\nAvg_Delivery_Days median:",
    round(
        delivery_median,
        4
    )
)


# ============================================================
# CHECK AFTER IMPUTATION
# ============================================================

print("\nMissing values after imputation:")

print(
    X.isna().sum()
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(
    X.describe()
    .round(2)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print("\nTraining target:")

print(
    y_train.value_counts()
)

print("\nTesting target:")

print(
    y_test.value_counts()
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
# PREDICT PROBABILITIES
# ============================================================

lr_prob = (
    lr_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# BASELINE METRICS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL")
print("=" * 70)

default_pred = (
    lr_prob >= 0.50
).astype(int)


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            lr_prob
        ),
        4
    )
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_test,
            lr_prob
        ),
        4
    )
)

print(
    "Default threshold:",
    0.50
)

print(
    "Default precision:",
    round(
        precision_score(
            y_test,
            default_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Default recall:",
    round(
        recall_score(
            y_test,
            default_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Default F1:",
    round(
        f1_score(
            y_test,
            default_pred,
            zero_division=0
        ),
        4
    )
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

threshold_results = []


for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    predictions = (
        lr_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    predicted_positive = (
        predictions.sum()
    )

    threshold_results.append({
        "threshold": round(
            threshold,
            2
        ),

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "predicted_positive": (
            predicted_positive
        )
    })


# ============================================================
# THRESHOLD DATAFRAME
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)


threshold_df = (
    threshold_df
    .sort_values(
        "f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# BEST THRESHOLD
# ============================================================

best = threshold_df.iloc[0]


print("\n" + "=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

print(
    "Threshold :",
    best["threshold"]
)

print(
    "Precision :",
    round(
        best["precision"],
        4
    )
)

print(
    "Recall    :",
    round(
        best["recall"],
        4
    )
)

print(
    "F1        :",
    round(
        best["f1"],
        4
    )
)

print(
    "Predicted Positive:",
    int(
        best[
            "predicted_positive"
        ]
    )
)


# ============================================================
# TOP 20 THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 THRESHOLDS")
print("=" * 70)

print(
    threshold_df
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# SPECIFIC THRESHOLD CHECK - 0.66
# ============================================================

selected_threshold = 0.66

selected_pred = (
    lr_prob >= selected_threshold
).astype(int)


selected_precision = precision_score(
    y_test,
    selected_pred,
    zero_division=0
)

selected_recall = recall_score(
    y_test,
    selected_pred,
    zero_division=0
)

selected_f1 = f1_score(
    y_test,
    selected_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("THRESHOLD 0.66 CHECK")
print("=" * 70)

print(
    "Threshold :",
    selected_threshold
)

print(
    "Precision :",
    round(
        selected_precision,
        4
    )
)

print(
    "Recall    :",
    round(
        selected_recall,
        4
    )
)

print(
    "F1        :",
    round(
        selected_f1,
        4
    )
)

print(
    "Predicted Positive:",
    int(
        selected_pred.sum()
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

threshold_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 70)

print(
    "\nSaved:",
    OUTPUT_PATH
)