# =========================================================
# WEEK 1 - LOGISTICS DATA ANALYSIS PROJECT
# E-COMMERCE DELIVERY AND INVENTORY OPTIMIZATION
# =========================================================

# Dataset:
# DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS
#
# Main objectives:
# 1. Understand the dataset
# 2. Check data quality
# 3. Calculate delivery KPIs
# 4. Analyse late-delivery risk
# 5. Analyse sales by product category
# 6. Analyse monthly demand
# 7. Segment customers using K-Means clustering
#
# =========================================================


# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------

FILE = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(FILE, encoding="latin1")

print("\n================================================")
print("DATASET INFORMATION")
print("================================================")

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ---------------------------------------------------------
# 3. VIEW FIRST FEW RECORDS
# ---------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------
# 4. DISPLAY COLUMN NAMES
# ---------------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 5. DATA TYPES
# ---------------------------------------------------------

print("\nData types:")
print(df.dtypes)


# ---------------------------------------------------------
# 6. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\n================================================")
print("DATA QUALITY CHECK")
print("================================================")

missing_values = (
    df.isna()
      .sum()
      .sort_values(ascending=False)
)

print("\nMissing values:")
print(missing_values.head(20))


# ---------------------------------------------------------
# 7. CHECK DUPLICATE ROWS
# ---------------------------------------------------------

duplicate_rows = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_rows)


# ---------------------------------------------------------
# 8. DATE PREPARATION
# ---------------------------------------------------------

# These are the actual date column names
# present in the DataCo dataset.

order_date_column = "order date (DateOrders)"
shipping_date_column = "shipping date (DateOrders)"

df[order_date_column] = pd.to_datetime(
    df[order_date_column],
    errors="coerce"
)

df[shipping_date_column] = pd.to_datetime(
    df[shipping_date_column],
    errors="coerce"
)

# Create useful time variables

df["Order Year"] = df[order_date_column].dt.year

df["Order Month"] = df[order_date_column].dt.month

df["Order Month Period"] = (
    df[order_date_column].dt.to_period("M")
)


# ---------------------------------------------------------
# 9. DELIVERY KPI ANALYSIS
# ---------------------------------------------------------

print("\n================================================")
print("DELIVERY KPI ANALYSIS")
print("================================================")

real_shipping_column = "Days for shipping (real)"

scheduled_shipping_column = "Days for shipment (scheduled)"

# Calculate difference between actual and scheduled time

df["Delay Days"] = (
    df[real_shipping_column]
    - df[scheduled_shipping_column]
)


# On-time delivery rate

on_time_rate = (
    (df["Delay Days"] <= 0).mean()
    * 100
)


# Average actual delivery/shipping time

average_delivery_time = (
    df[real_shipping_column].mean()
)


print(
    "\nOn-time delivery rate:",
    round(on_time_rate, 2),
    "%"
)

print(
    "Average delivery time:",
    round(average_delivery_time, 2),
    "days"
)


# ---------------------------------------------------------
# 10. LATE DELIVERY RISK BY SHIPPING MODE
# ---------------------------------------------------------

print("\n================================================")
print("LATE-DELIVERY RISK BY SHIPPING MODE")
print("================================================")

late_by_mode = (
    df.groupby("Shipping Mode")["Late_delivery_risk"]
      .mean()
      .sort_values(ascending=False)
      * 100
)

print(late_by_mode.round(2))


# Save the result

late_by_mode.round(2).to_csv(
    "late_delivery_by_shipping_mode.csv"
)


# Create graph

plt.figure(figsize=(10, 6))

late_by_mode.plot(kind="bar")

plt.title(
    "Late-Delivery Risk by Shipping Mode"
)

plt.xlabel("Shipping Mode")

plt.ylabel(
    "Late-Delivery Risk (%)"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "late_delivery_by_shipping_mode.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 11. SALES BY PRODUCT CATEGORY
# ---------------------------------------------------------

print("\n================================================")
print("TOP PRODUCT CATEGORIES BY SALES")
print("================================================")

category_sales = (
    df.groupby("Category Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print(
    category_sales.head(10)
)


# Save category sales

category_sales.to_csv(
    "category_sales.csv"
)


# Create graph

plt.figure(figsize=(10, 6))

category_sales.head(10).plot(
    kind="bar"
)

plt.title(
    "Top 10 Product Categories by Sales"
)

plt.xlabel("Product Category")

plt.ylabel("Total Sales")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    "top_categories_sales.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 12. MONTHLY DEMAND ANALYSIS
# ---------------------------------------------------------

print("\n================================================")
print("MONTHLY DEMAND ANALYSIS")
print("================================================")

monthly_demand = (
    df.groupby("Order Month Period")[
        "Order Item Quantity"
    ]
    .sum()
)


print("\nMonthly demand:")
print(monthly_demand)


# Save monthly demand

monthly_demand.to_csv(
    "monthly_demand.csv"
)


# Create graph

plt.figure(figsize=(12, 6))

monthly_demand.plot()

plt.title(
    "Monthly Order Quantity"
)

plt.xlabel("Month")

plt.ylabel(
    "Order Item Quantity"
)

plt.tight_layout()

plt.savefig(
    "monthly_demand.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 13. TOP PRODUCTS BY QUANTITY
# ---------------------------------------------------------

print("\n================================================")
print("TOP PRODUCTS BY QUANTITY SOLD")
print("================================================")

product_quantity = (
    df.groupby("Product Name")[
        "Order Item Quantity"
    ]
    .sum()
    .sort_values(ascending=False)
)

print(
    product_quantity.head(10)
)


# Save results

product_quantity.head(20).to_csv(
    "top_products_by_quantity.csv"
)


# ---------------------------------------------------------
# 14. CUSTOMER CLUSTERING USING K-MEANS
# ---------------------------------------------------------

print("\n================================================")
print("CUSTOMER CLUSTERING")
print("================================================")


# Create customer-level information

customer_features = (
    df.groupby("Customer Id")
      .agg(
          total_orders=(
              "Order Id",
              "nunique"
          ),

          total_quantity=(
              "Order Item Quantity",
              "sum"
          ),

          total_sales=(
              "Sales",
              "sum"
          )
      )
      .reset_index()
)


print("\nNumber of customers analysed:")

print(
    len(customer_features)
)


# Select features for clustering

X = customer_features[
    [
        "total_orders",
        "total_quantity",
        "total_sales"
    ]
].fillna(0)


# Standardize the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Create K-Means model

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Assign each customer to a cluster

customer_features["cluster"] = (
    model.fit_predict(X_scaled)
)


# ---------------------------------------------------------
# 15. CUSTOMER CLUSTER COUNTS
# ---------------------------------------------------------

print("\nCustomer cluster counts:")

cluster_counts = (
    customer_features["cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


# Save cluster counts

cluster_counts.to_csv(
    "customer_cluster_counts.csv"
)


# ---------------------------------------------------------
# 16. CUSTOMER CLUSTER PROFILES
# ---------------------------------------------------------

print("\nCustomer cluster profiles:")

cluster_profile = (
    customer_features
    .groupby("cluster")[
        [
            "total_orders",
            "total_quantity",
            "total_sales"
        ]
    ]
    .mean()
    .round(2)
)

print(cluster_profile)


# Save cluster profiles

cluster_profile.to_csv(
    "customer_cluster_profiles.csv"
)


# Save complete customer dataset with clusters

customer_features.to_csv(
    "customer_clusters.csv",
    index=False
)


# ---------------------------------------------------------
# 17. CREATE CLUSTER VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    customer_features["total_orders"],
    customer_features["total_sales"],
    c=customer_features["cluster"]
)

plt.title(
    "Customer Segmentation Using K-Means"
)

plt.xlabel(
    "Total Orders"
)

plt.ylabel(
    "Total Sales"
)

plt.tight_layout()

plt.savefig(
    "customer_clusters.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 18. FINAL SUMMARY
# ---------------------------------------------------------

print("\n================================================")
print("FINAL ANALYSIS SUMMARY")
print("================================================")

print(
    "\nDataset size:",
    df.shape[0],
    "rows and",
    df.shape[1],
    "columns"
)

print(
    "Duplicate rows:",
    duplicate_rows
)

print(
    "On-time delivery rate:",
    round(on_time_rate, 2),
    "%"
)

print(
    "Average delivery time:",
    round(average_delivery_time, 2),
    "days"
)

print(
    "\nHighest late-delivery-risk shipping mode:"
)

print(
    late_by_mode.index[0],
    "-",
    round(late_by_mode.iloc[0], 2),
    "%"
)

print(
    "\nLowest late-delivery-risk shipping mode:"
)

print(
    late_by_mode.index[-1],
    "-",
    round(late_by_mode.iloc[-1], 2),
    "%"
)

print(
    "\nHighest-sales product category:"
)

print(
    category_sales.index[0],
    "-",
    round(category_sales.iloc[0], 2)
)

print(
    "\nNumber of customer clusters:",
    customer_features["cluster"].nunique()
)


print("\n================================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("================================================")

print(
    "\nFiles and graphs have been saved "
    "inside your project folder."
)