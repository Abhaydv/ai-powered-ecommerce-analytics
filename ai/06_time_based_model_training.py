# TIME-BASED REPEAT PURCHASE MODEL TRAININ
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
# 1. PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_PATH = (
    BASE_DIR / "data" / "processed"
    / "master_ecommerce_dataset.csv"
)

ORDER_ITEMS_PATH = (
    BASE_DIR / "data" / "processed"
    / "order_items_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. SETTINGS

FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Avg_Order_Value",
    "Total_Items",
    "Customer_Lifetime_Days",
    "Items_Per_Order"
]

# Historical training snapshots.
TRAIN_CUTOFFS = [
    "2017-04-01",
    "2017-05-01",
    "2017-06-01"
]

VALIDATION_CUTOFF = "2017-12-01"
TEST_CUTOFF = "2018-06-01"

OBSERVATION_MONTHS = 3
TARGET_MONTHS = 3

RANDOM_STATE = 42

# 3. LOAD DATA

print("\n" + "=" * 70)
print("TIME-BASED REPEAT PURCHASE MODEL")
print("=" * 70)

print("\nLoading master dataset...")

master_df = pd.read_csv(MASTER_PATH)

master_df["order_purchase_timestamp"] = pd.to_datetime(
    master_df["order_purchase_timestamp"],
    errors="coerce"
)

print("Master rows:", len(master_df))


print("\nLoading order items...")

order_items = pd.read_csv(ORDER_ITEMS_PATH)

print("Order item rows:", len(order_items))

# 4. CORRECT ORDER-LEVEL AGGREGATION

print("\nAggregating item revenue at order level...")

orders = (
    master_df
    .groupby("order_id", as_index=False)
    .agg(
        customer_unique_id=(
            "customer_unique_id", "first"
        ),

        order_purchase_timestamp=(
            "order_purchase_timestamp", "first"
        ),

        order_revenue=(
            "order_revenue", "sum"
        )
    )
)

# Calculate actual item count per order.
item_counts = (
    order_items
    .groupby("order_id")
    .size()
    .reset_index(name="Total_Items")
)

orders = orders.merge(
    item_counts,
    on="order_id",
    how="left"
)

orders["Total_Items"] = (
    orders["Total_Items"].fillna(0)
)

orders["order_revenue"] = (
    orders["order_revenue"].fillna(0)
)

orders = orders.dropna(
    subset=[
        "customer_unique_id",
        "order_purchase_timestamp"
    ]
)

orders = orders.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)

print("Unique orders:", orders["order_id"].nunique())
print("Unique customers:", orders["customer_unique_id"].nunique())


# 5. CUSTOMER FEATURE ENGINEERING

def create_cohort(cutoff_date, cohort_name):

    cutoff = pd.Timestamp(cutoff_date)

    observation_start = (
        cutoff - pd.DateOffset(
            months=OBSERVATION_MONTHS
        )
    )

    target_end = (
        cutoff + pd.DateOffset(
            months=TARGET_MONTHS
        )
    )

    print("\n" + "-" * 65)
    print(f"CREATING {cohort_name.upper()} COHORT")
    print("-" * 65)

    # Customers active during the observation window.
    observation_orders = orders[
        (
            orders["order_purchase_timestamp"]
            >= observation_start
        )
        &
        (
            orders["order_purchase_timestamp"]
            < cutoff
        )
    ]

    active_customers = set(
        observation_orders[
            "customer_unique_id"
        ].unique()
    )

    # All historical orders available before cutoff.
    historical_orders = orders[
        orders["order_purchase_timestamp"] < cutoff
    ].copy()

    # Future orders within the target window.
    future_orders = orders[
        (
            orders["order_purchase_timestamp"]
            >= cutoff
        )
        &
        (
            orders["order_purchase_timestamp"]
            < target_end
        )
    ]

    future_customers = set(
        future_orders[
            "customer_unique_id"
        ].unique()
    )

    # Customer-level features.
    features = (
        historical_orders
        .groupby("customer_unique_id")
        .agg(
            Frequency=(
                "order_id", "nunique"
            ),

            Monetary=(
                "order_revenue", "sum"
            ),

            Avg_Order_Value=(
                "order_revenue", "mean"
            ),

            Total_Items=(
                "Total_Items", "sum"
            ),

            Last_Purchase=(
                "order_purchase_timestamp", "max"
            ),

            First_Purchase=(
                "order_purchase_timestamp", "min"
            )
        )
        .reset_index()
    )

    # Only retain active customers.
    features = features[
        features["customer_unique_id"].isin(
            active_customers
        )
    ].copy()

    # Recency.
    features["Recency"] = (
        cutoff - features["Last_Purchase"]
    ).dt.days

    # Customer lifetime.
    features["Customer_Lifetime_Days"] = (
        features["Last_Purchase"]
        - features["First_Purchase"]
    ).dt.days

    # Items per order.
    features["Items_Per_Order"] = (
        features["Total_Items"]
        / features["Frequency"].replace(0, np.nan)
    )

    # Future purchase target.
    features["repeat_purchase"] = (
        features["customer_unique_id"]
        .isin(future_customers)
        .astype(int)
    )

    # Final cohort.
    features = features[
        [
            "customer_unique_id",
            *FEATURE_COLS,
            "repeat_purchase"
        ]
    ]

    print("Observation orders:", len(observation_orders))
    print("Future orders:", len(future_orders))
    print("Active customers:", len(features))
    print(
        "Repeat customers:",
        int(features["repeat_purchase"].sum())
    )

    repeat_rate = (
        features["repeat_purchase"].mean() * 100
    )

    print("Repeat rate:", round(repeat_rate, 2), "%")

    return features


# 6. BUILD TRAINING DATA


print("\nBuilding historical training cohorts...")

training_cohorts = []

for cutoff_date in TRAIN_CUTOFFS:

    cohort = create_cohort(
        cutoff_date,
        f"Training {cutoff_date}"
    )

    cohort["snapshot_date"] = cutoff_date

    training_cohorts.append(cohort)


train_df = pd.concat(
    training_cohorts,
    ignore_index=True
)

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df["repeat_purchase"].copy()

print("\nCombined training dataset:", X_train.shape)
print("\nTraining target distribution:")
print(y_train.value_counts())

# 7. VALIDATION DATA

validation_df = create_cohort(
    VALIDATION_CUTOFF,
    "Validation"
)

X_val = validation_df[FEATURE_COLS].copy()
y_val = validation_df["repeat_purchase"].copy()

# 8. TEST DATA


test_df = create_cohort(
    TEST_CUTOFF,
    "Test"
)

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df["repeat_purchase"].copy()

# 9. DATA QUALITY CHECK


print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test)
]:

    print(f"\n{name}")
    print("Shape:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))
    print("Positive samples:", int(y.sum()))
    print("Negative samples:", int((y == 0).sum()))

    if y.nunique() < 2:
        raise ValueError(
            f"{name} dataset contains only one target class."
        )

# 10. MODEL DEFINITIONS

models = {

    "Logistic Regression": Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])
}

# 11. TRAIN AND VALIDATE MODELS


validation_results = []
trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND VALIDATION")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    validation_results.append({
        "Model": model_name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    trained_models[model_name] = model

    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))


# 12. SELECT BEST MODEL

validation_results_df = pd.DataFrame(validation_results)

validation_results_df = validation_results_df.sort_values(
    "PR_AUC", ascending=False
)

print("\nValidation Model Comparison:")
print(validation_results_df.to_string(index=False))

best_model_name = validation_results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nSelected Model:", best_model_name)


# 13. THRESHOLD OPTIMIZATION

val_probabilities = best_model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_probabilities)
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    /
    (
        precision_values[:-1] +
        recall_values[:-1] + 1e-10
    )
)

best_index = np.argmax(f1_values)

selected_threshold = float(thresholds[best_index])

print("\nSelected Threshold:", round(selected_threshold, 4))


# 14. FINAL TEST EVALUATION

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

print("\nFINAL TEST RESULTS")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)

test_metrics = {
    "Selected_Model": best_model_name,
    "Threshold": selected_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC_AUC": roc_auc_score(y_test, test_probabilities),
    "PR_AUC": average_precision_score(
        y_test, test_probabilities
    )
}

print("\nTest Metrics:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value}")


# 15. SAVE MODELS

joblib.dump(
    best_model,
    MODEL_DIR / "best_repeat_purchase_model_v2.pkl"
)


# 16. SAVE RESULTS

validation_results_df.to_csv(
    OUTPUT_DIR / "model_validation_comparison_v2.csv",
    index=False
)

pd.DataFrame([test_metrics]).to_csv(
    OUTPUT_DIR / "model_test_metrics_v2.csv",
    index=False
)

# 17. SAVE PREDICTIONS

predictions_df = pd.DataFrame({
    "customer_unique_id": test_df["customer_unique_id"].values,
    "actual_repeat_purchase": y_test.values,
    "repeat_purchase_probability": test_probabilities,
    "predicted_repeat_purchase": test_predictions
})

predictions_df["priority"] = np.select(
    [
        test_probabilities >= selected_threshold,
        test_probabilities >= 0.50
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)

predictions_df.to_csv(
    OUTPUT_DIR / "repeat_purchase_test_predictions_v2.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("\nPriority Distribution:")
print(predictions_df["priority"].value_counts())