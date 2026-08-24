# MODEL TRAINING - REPEAT PURCHASE PREDICTION

import pandas as pd
import joblib

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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LR_MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression.pkl"
)

RF_MODEL_PATH = (
    MODEL_DIR
    / "random_forest.pkl"
)

METADATA_PATH = (
    MODEL_DIR
    / "model_metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CUTOFF_DATE = pd.Timestamp(
    "2018-03-01"
)

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
# LOAD MASTER DATASET
# ============================================================

print("=" * 70)
print("TIME-BASED REPEAT PURCHASE MODEL TRAINING")
print("=" * 70)

print("\nLoading master dataset...")

master_df = pd.read_csv(
    MASTER_PATH
)

master_df[
    "order_purchase_timestamp"
] = pd.to_datetime(
    master_df[
        "order_purchase_timestamp"
    ],
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
# REQUIRED COLUMNS
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


# ============================================================
# ORDER-LEVEL DATA
# ============================================================

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


orders[
    "Total_Items"
] = (
    orders[
        "Total_Items"
    ]
    .fillna(0)
)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

past_orders = orders[
    orders[
        "order_purchase_timestamp"
    ] < CUTOFF_DATE
].copy()


future_orders = orders[
    orders[
        "order_purchase_timestamp"
    ] >= CUTOFF_DATE
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
# CUSTOMER FEATURES FROM PAST DATA
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

customer_features[
    "Recency"
] = (
    CUTOFF_DATE
    - customer_features[
        "Last_Purchase"
    ]
).dt.days


# ============================================================
# CUSTOMER LIFETIME
# ============================================================

customer_features[
    "Customer_Lifetime_Days"
] = (
    customer_features[
        "Last_Purchase"
    ]
    -
    customer_features[
        "First_Purchase"
    ]
).dt.days


# ============================================================
# ITEMS PER ORDER
# ============================================================

customer_features[
    "Items_Per_Order"
] = (
    customer_features[
        "Total_Items"
    ]
    /
    customer_features[
        "Frequency"
    ]
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
# FINAL ML DATASET
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

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    X.isna().sum()
)


delivery_median = (
    X[
        "Avg_Delivery_Days"
    ].median()
)


X[
    "Avg_Delivery_Days"
] = (
    X[
        "Avg_Delivery_Days"
    ]
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
# FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ML DATASET")
print("=" * 70)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print("\nFeatures:")

print(
    FEATURE_COLS
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

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


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
# LOGISTIC REGRESSION PREDICTIONS
# ============================================================

lr_pred = lr_model.predict(
    X_test
)


lr_prob = (
    lr_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# LOGISTIC RESULTS
# ============================================================

lr_roc_auc = roc_auc_score(
    y_test,
    lr_prob
)

lr_pr_auc = average_precision_score(
    y_test,
    lr_prob
)


print("\nClassification Report:")

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


print(
    "\nROC-AUC:",
    round(
        lr_roc_auc,
        4
    )
)


print(
    "PR-AUC:",
    round(
        lr_pr_auc,
        4
    )
)


# ============================================================
# SAVE LOGISTIC MODEL
# ============================================================

joblib.dump(
    lr_model,
    LR_MODEL_PATH
)


print(
    "\nLogistic Regression saved to:"
)

print(
    LR_MODEL_PATH
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


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


# ============================================================
# RANDOM FOREST PREDICTIONS
# ============================================================

rf_pred = rf_model.predict(
    X_test
)


rf_prob = (
    rf_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# RANDOM FOREST RESULTS
# ============================================================

rf_roc_auc = roc_auc_score(
    y_test,
    rf_prob
)

rf_pr_auc = average_precision_score(
    y_test,
    rf_prob
)


print("\nClassification Report:")

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


print(
    "\nROC-AUC:",
    round(
        rf_roc_auc,
        4
    )
)


print(
    "PR-AUC:",
    round(
        rf_pr_auc,
        4
    )
)


# ============================================================
# SAVE RANDOM FOREST
# ============================================================

joblib.dump(
    rf_model,
    RF_MODEL_PATH
)


print(
    "\nRandom Forest saved to:"
)

print(
    RF_MODEL_PATH
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature":
        FEATURE_COLS,

    "Importance":
        rf_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance
    .to_string(
        index=False
    )
)


# ============================================================
# MODEL METADATA
# ============================================================

model_metadata = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "ROC_AUC": [
        lr_roc_auc,
        rf_roc_auc
    ],

    "PR_AUC": [
        lr_pr_auc,
        rf_pr_auc
    ],

    "Cutoff_Date": [
        CUTOFF_DATE,
        CUTOFF_DATE
    ],

    "Training_Samples": [
        len(X_train),
        len(X_train)
    ],

    "Testing_Samples": [
        len(X_test),
        len(X_test)
    ]
})


model_metadata.to_csv(
    METADATA_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nSaved models:")

print(
    LR_MODEL_PATH
)

print(
    RF_MODEL_PATH
)

print("\nSaved metadata:")

print(
    METADATA_PATH
)

print("\nModel comparison:")

print(
    model_metadata.round(
        4
    ).to_string(
        index=False
    )
)