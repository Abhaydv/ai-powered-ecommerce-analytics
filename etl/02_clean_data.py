import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    path = RAW_DIR / filename
    return pd.read_csv(path)


print("=" * 70)
print("STARTING E-COMMERCE DATA CLEANING")
print("=" * 70)


# --------------------------------------------------
# 1. CUSTOMERS
# --------------------------------------------------

customers = load_csv("olist_customers_dataset.csv")

customers = customers.drop_duplicates()

customers.to_csv(
    PROCESSED_DIR / "customers_clean.csv",
    index=False
)

print("✓ Customers cleaned")


# --------------------------------------------------
# 2. ORDERS
# --------------------------------------------------

orders = load_csv("olist_orders_dataset.csv")

orders = orders.drop_duplicates()

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )


# Actual delivery time in days
orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400


# Difference between actual and estimated delivery
orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400


# Delivery status
orders["is_delayed"] = (
    orders["delivery_delay_days"] > 0
)

orders.to_csv(
    PROCESSED_DIR / "orders_clean.csv",
    index=False
)

print("✓ Orders cleaned")


# --------------------------------------------------
# 3. ORDER ITEMS
# --------------------------------------------------

order_items = load_csv("olist_order_items_dataset.csv")

order_items = order_items.drop_duplicates()

order_items["shipping_limit_date"] = pd.to_datetime(
    order_items["shipping_limit_date"],
    errors="coerce"
)

# Total item value
order_items["item_total_value"] = (
    order_items["price"]
    + order_items["freight_value"]
)

order_items.to_csv(
    PROCESSED_DIR / "order_items_clean.csv",
    index=False
)

print("✓ Order items cleaned")


# --------------------------------------------------
# 4. PAYMENTS
# --------------------------------------------------

payments = load_csv("olist_order_payments_dataset.csv")

payments = payments.drop_duplicates()

payments["payment_type"] = (
    payments["payment_type"]
    .fillna("unknown")
)

payments.to_csv(
    PROCESSED_DIR / "payments_clean.csv",
    index=False
)

print("✓ Payments cleaned")


# --------------------------------------------------
# 5. REVIEWS
# --------------------------------------------------

reviews = load_csv("olist_order_reviews_dataset.csv")

reviews = reviews.drop_duplicates()

reviews["review_comment_title"] = (
    reviews["review_comment_title"]
    .fillna("")
)

reviews["review_comment_message"] = (
    reviews["review_comment_message"]
    .fillna("")
)

reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"],
    errors="coerce"
)

reviews["review_answer_timestamp"] = pd.to_datetime(
    reviews["review_answer_timestamp"],
    errors="coerce"
)

# Sentiment category based on score
reviews["sentiment"] = reviews["review_score"].apply(
    lambda x:
        "Positive" if x >= 4
        else "Neutral" if x == 3
        else "Negative"
)

reviews.to_csv(
    PROCESSED_DIR / "reviews_clean.csv",
    index=False
)

print("✓ Reviews cleaned")


# --------------------------------------------------
# 6. PRODUCTS
# --------------------------------------------------

products = load_csv("olist_products_dataset.csv")

products = products.drop_duplicates()

# Missing category
products["product_category_name"] = (
    products["product_category_name"]
    .fillna("unknown")
)

# Numerical columns
numeric_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for col in numeric_columns:
    products[col] = products[col].fillna(
        products[col].median()
    )

products.to_csv(
    PROCESSED_DIR / "products_clean.csv",
    index=False
)

print("✓ Products cleaned")


# --------------------------------------------------
# 7. SELLERS
# --------------------------------------------------

sellers = load_csv("olist_sellers_dataset.csv")

sellers = sellers.drop_duplicates()

sellers["seller_city"] = (
    sellers["seller_city"]
    .fillna("unknown")
)

sellers["seller_state"] = (
    sellers["seller_state"]
    .fillna("unknown")
)

sellers.to_csv(
    PROCESSED_DIR / "sellers_clean.csv",
    index=False
)

print("✓ Sellers cleaned")


# --------------------------------------------------
# 8. PRODUCT CATEGORY TRANSLATION
# --------------------------------------------------

categories = load_csv(
    "product_category_name_translation.csv"
)

categories = categories.drop_duplicates()

categories.to_csv(
    PROCESSED_DIR / "category_translation_clean.csv",
    index=False
)

print("✓ Category translation cleaned")


print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nProcessed files:")

for file in PROCESSED_DIR.glob("*.csv"):
    print(f"✓ {file.name}")