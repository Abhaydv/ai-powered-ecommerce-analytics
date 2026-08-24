import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load(name):
    return pd.read_csv(PROCESSED_DIR / name)


print("=" * 70)
print("CREATING MASTER E-COMMERCE DATASET")
print("=" * 70)


# Load datasets
orders = load("orders_clean.csv")
order_items = load("order_items_clean.csv")
payments = load("payments_clean.csv")
customers = load("customers_clean.csv")
products = load("products_clean.csv")
sellers = load("sellers_clean.csv")
reviews = load("reviews_clean.csv")
categories = load("category_translation_clean.csv")


# --------------------------------------------------
# ORDERS + CUSTOMERS
# --------------------------------------------------

master = orders.merge(
    customers,
    on="customer_id",
    how="left"
)


# --------------------------------------------------
# ORDER ITEMS
# --------------------------------------------------

master = master.merge(
    order_items,
    on="order_id",
    how="left"
)


# --------------------------------------------------
# PRODUCTS
# --------------------------------------------------

master = master.merge(
    products,
    on="product_id",
    how="left"
)


# --------------------------------------------------
# CATEGORY TRANSLATION
# --------------------------------------------------

master = master.merge(
    categories,
    on="product_category_name",
    how="left"
)


# --------------------------------------------------
# SELLERS
# --------------------------------------------------

master = master.merge(
    sellers,
    on="seller_id",
    how="left"
)


# --------------------------------------------------
# PAYMENTS
# --------------------------------------------------

payment_summary = (
    payments
    .groupby("order_id")
    .agg(
        total_payment_value=("payment_value", "sum"),
        payment_count=("payment_sequential", "count"),
        avg_installments=("payment_installments", "mean")
    )
    .reset_index()
)

master = master.merge(
    payment_summary,
    on="order_id",
    how="left"
)


# --------------------------------------------------
# REVIEWS
# --------------------------------------------------

review_summary = (
    reviews
    .groupby("order_id")
    .agg(
        review_score=("review_score", "mean")
    )
    .reset_index()
)

master = master.merge(
    review_summary,
    on="order_id",
    how="left"
)


# --------------------------------------------------
# BUSINESS FEATURES
# --------------------------------------------------

master["order_revenue"] = (
    master["price"].fillna(0)
    + master["freight_value"].fillna(0)
)

master["order_month"] = (
    master["order_purchase_timestamp"]
    .astype(str)
    .str[:7]
)

master["order_year"] = (
    master["order_purchase_timestamp"]
    .astype(str)
    .str[:4]
)

master["product_category_name_english"] = (
    master["product_category_name_english"]
    .fillna("unknown")
)

master["review_score"] = (
    master["review_score"]
    .fillna(0)
)


# --------------------------------------------------
# SAVE MASTER DATASET
# --------------------------------------------------

output_path = PROCESSED_DIR / "master_ecommerce_dataset.csv"

master.to_csv(
    output_path,
    index=False
)


print("\n✓ Master dataset created")
print(f"Rows    : {master.shape[0]}")
print(f"Columns : {master.shape[1]}")
print(f"Saved   : {output_path}")

print("\nSample columns:")
print(master.columns.tolist())

print("\nMissing values:")
print(
    master.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

print("\n" + "=" * 70)
print("MASTER DATASET CREATION COMPLETED")
print("=" * 70)