"""
data_processing.py
-------------------
Reusable functions for loading and cleaning the Telco Customer Churn dataset.

Author: Aptura Tech Solutions - Data Science Internship (Batch 02, Week 03)
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Fix TotalCharges (stored as string with blank values for new customers)
      - Drop the non-predictive customerID column
      - Standardize the target column to a binary integer
      - Remove exact duplicate rows
    """
    df = df.copy()

    # TotalCharges has blank strings for customers with tenure == 0.
    # Coerce to numeric; blanks become NaN, then impute with 0 (no charges yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop identifier column - has no predictive value and would leak
    # a unique-per-row key into the model if accidentally encoded.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Standardize target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows.")

    return df


def get_feature_types(df: pd.DataFrame, target: str = "Churn"):
    """Split columns into numeric and categorical feature lists (excludes target)."""
    features = [c for c in df.columns if c != target]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]
    return numeric_features, categorical_features
