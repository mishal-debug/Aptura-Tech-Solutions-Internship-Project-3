"""
feature_engineering.py
-----------------------
Feature engineering and preprocessing pipeline construction for the
Telco Customer Churn predictive model.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional business-meaningful features from raw columns.

    New features:
      - tenure_years        : tenure expressed in years (easier business interpretation)
      - avg_monthly_spend   : TotalCharges / tenure (guards against div-by-zero)
      - num_services        : count of subscribed add-on services
      - is_new_customer     : flag for tenure <= 3 months (highest churn-risk segment)
      - charge_per_service  : MonthlyCharges relative to number of services subscribed
    """
    df = df.copy()

    df["tenure_years"] = df["tenure"] / 12.0

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies",
    ]

    def count_services(row):
        count = 0
        for col in service_cols:
            val = row[col]
            if val not in ("No", "No internet service", "No phone service"):
                count += 1
        return count

    df["num_services"] = df.apply(count_services, axis=1)

    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    df["charge_per_service"] = df["MonthlyCharges"] / df["num_services"].replace(0, 1)

    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - scales numeric features (zero mean, unit variance)
      - one-hot encodes categorical features (drop first level to avoid collinearity)
    This is fit only on the TRAINING split to prevent data leakage.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor
