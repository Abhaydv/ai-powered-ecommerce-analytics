# MODEL EVALUATION

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
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
    / "model_comparison.csv"
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("\nLoading datasets...")

master_df = pd.read_csv(MASTER_PATH)

master_df["order_purchase_timestamp"] = pd.to_datetime(
    master_df["order_purchase_timestamp"],
    errors="coerce"
)

order_items = pd.read_csv(
    ORDER_ITEMS_PATH
)


# ============================================================
# ITEM COUNTS
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
        f"Missing columns: {missing_columns}"
    )


orders = (
    master_df[required_columns]
    .drop_duplicates("order_id")
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
# TIME-BASED DATA
# ============================================================

past_orders = orders[
    orders["order_purchase_timestamp"] < CUTOFF_DATE
].copy()

future_orders = orders[
    orders["order_purchase_timestamp"] >= CUTOFF_DATE
].copy()


future_customers = set(
    future_orders[
        "customer_unique_id"
    ]
    .dropna()
    .unique()
)


print("\nCutoff Date:", CUTOFF_DATE)
print("Past Orders:", len(past_orders))
print("Future Orders:", len(future_orders))
print(
    "Future purchasing customers:",
    len(future_customers)
)


# ============================================================
# CUSTOMER FEATURES
# ============================================================

customer_features = (
    past_orders
    .groupby("customer_unique_id")
    .agg(
        Frequency=("order_id", "nunique"),

        Monetary=("order_revenue", "sum"),

        Avg_Order_Value=("order_revenue", "mean"),

        Avg_Review_Score=("review_score", "mean"),

        Avg_Delivery_Days=("delivery_days", "mean"),

        Total_Items=("Total_Items", "sum"),

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


customer_features["Recency"] = (
    CUTOFF_DATE
    - customer_features["Last_Purchase"]
).dt.days


customer_features["Customer_Lifetime_Days"] = (
    customer_features["Last_Purchase"]
    - customer_features["First_Purchase"]
).dt.days


customer_features["Items_Per_Order"] = (
    customer_features["Total_Items"]
    / customer_features["Frequency"]
)


customer_features["repeat_purchase"] = (
    customer_features["customer_unique_id"]
    .isin(future_customers)
    .astype(int)
)


# ============================================================
# X / y
# ============================================================

X = customer_features[
    FEATURE_COLS
].copy()

y = customer_features[
    "repeat_purchase"
].copy()


# ============================================================
# MISSING VALUES
# ============================================================

if X["Avg_Delivery_Days"].isna().sum() > 0:

    median_delivery = (
        X["Avg_Delivery_Days"]
        .median()
    )

    X["Avg_Delivery_Days"] = (
        X["Avg_Delivery_Days"]
        .fillna(median_delivery)
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
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

lr_pred = lr_model.predict(
    X_test
)

lr_prob = lr_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# FUNCTION FOR MODEL METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    return {
        "Accuracy": (
            predictions == y_true
        ).mean(),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# CALCULATE RESULTS
# ============================================================

lr_metrics = calculate_metrics(
    y_test,
    lr_pred,
    lr_prob
)

rf_metrics = calculate_metrics(
    y_test,
    rf_pred,
    rf_prob
)


# ============================================================
# MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        **lr_metrics
    },
    {
        "Model": "Random Forest",
        **rf_metrics
    }
])


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    model_comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# LOGISTIC REGRESSION REPORT
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

print(
    classification_report(
        y_test,
        lr_pred,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        lr_pred
    )
)


# ============================================================
# RANDOM FOREST REPORT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

print(
    classification_report(
        y_test,
        rf_pred,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

model_comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_PATH)