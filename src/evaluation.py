"""
evaluation.py
--------------
Evaluation metrics, plots, and reporting utilities for classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)


def evaluate_model(model, X_train, y_train, X_test, y_test, name="model"):
    """Compute train/test accuracy plus a full battery of test-set metrics."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results = {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1_score": f1_score(y_test, test_pred),
        "roc_auc": roc_auc_score(y_test, test_proba) if test_proba is not None else np.nan,
    }
    results["train_test_gap"] = results["train_accuracy"] - results["test_accuracy"]
    return results


def plot_confusion_matrix(y_test, y_pred, ax=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curve(y_test, y_proba, ax=None, label="model"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_feature_importance(model, feature_names, top_n=15, ax=None):
    """Plot top-N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="#2E7D6B")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    return ax


def get_feature_names(preprocessor, numeric_features, categorical_features):
    """Retrieve feature names after ColumnTransformer preprocessing."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return numeric_features + cat_names
