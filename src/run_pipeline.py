"""
run_pipeline.py
----------------
End-to-end execution: load -> clean -> engineer -> split -> train -> tune ->
evaluate -> save artifacts (model, metrics, plots) for the report/dashboard.
"""

import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_processing import load_data, clean_data, get_feature_types
from feature_engineering import engineer_features, build_preprocessor
from model_training import (
    get_candidate_models, get_param_grids, build_pipeline, tune_model, XGBOOST_AVAILABLE,
)
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, get_feature_names,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def main():
    t0 = time.time()
    print("=" * 70)
    print("APTURA TECH SOLUTIONS — Predictive Analytics Pipeline")
    print("Telco Customer Churn Prediction")
    print("=" * 70)

    # 1. DATA COLLECTION -----------------------------------------------
    df_raw = load_data("../data/telco.csv")
    print(f"\n[1] Data loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    # 2. DATA CLEANING ---------------------------------------------------
    df_clean = clean_data(df_raw)
    print(f"[2] Data cleaned: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    print(f"    Missing values remaining: {df_clean.isnull().sum().sum()}")
    print(f"    Churn rate: {df_clean['Churn'].mean():.2%}")

    # 3. FEATURE ENGINEERING ---------------------------------------------
    df_feat = engineer_features(df_clean)
    print(f"[3] Feature engineering complete: {df_feat.shape[1]} total columns")

    df_feat.to_csv("../data/telco_processed.csv", index=False)

    X = df_feat.drop(columns=["Churn"])
    y = df_feat["Churn"]

    numeric_features, categorical_features = get_feature_types(df_feat, target="Churn")
    print(f"    Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"    Categorical features ({len(categorical_features)}): {categorical_features}")

    # 4. TRAIN/TEST SPLIT -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[4] Split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 5. DEMONSTRATE OVERFITTING (Question 01 setup) ----------------------
    overfit_pipeline = build_pipeline(
        preprocessor,
        DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=RANDOM_STATE),
    )
    overfit_pipeline.fit(X_train, y_train)
    overfit_results = evaluate_model(
        overfit_pipeline, X_train, y_train, X_test, y_test, name="Unconstrained Decision Tree"
    )
    print(f"\n[Overfitting Demo] Unconstrained Decision Tree:")
    print(f"    Train accuracy: {overfit_results['train_accuracy']:.3f}")
    print(f"    Test accuracy:  {overfit_results['test_accuracy']:.3f}")
    print(f"    Gap:            {overfit_results['train_test_gap']:.3f}")

    # 6. MODEL TRAINING & COMPARISON --------------------------------------
    print("\n[5] Training candidate models (baseline hyperparameters)...")
    candidate_models = get_candidate_models()
    baseline_results = []
    fitted_baselines = {}
    for name, model in candidate_models.items():
        pipe = build_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)
        res = evaluate_model(pipe, X_train, y_train, X_test, y_test, name=name)
        baseline_results.append(res)
        fitted_baselines[name] = pipe
        print(f"    {name:<20s} train_acc={res['train_accuracy']:.3f}  "
              f"test_acc={res['test_accuracy']:.3f}  f1={res['f1_score']:.3f}  "
              f"auc={res['roc_auc']:.3f}")

    baseline_df = pd.DataFrame(baseline_results).sort_values("roc_auc", ascending=False)
    baseline_df.to_csv("../outputs/baseline_model_comparison.csv", index=False)

    # 7. HYPERPARAMETER TUNING (top 2 models by baseline AUC) -------------
    top_models = baseline_df["model"].head(2).tolist()
    print(f"\n[6] Hyperparameter tuning top models: {top_models}")
    param_grids = get_param_grids()
    tuned_results = []
    fitted_tuned = {}
    for name in top_models:
        if name not in param_grids:
            continue
        pipe = build_pipeline(preprocessor, candidate_models[name])
        search_type = "grid" if name in ("LogisticRegression", "DecisionTree") else "random"
        search = tune_model(
            pipe, param_grids[name], X_train, y_train,
            cv=5, search_type=search_type, n_iter=15, scoring="roc_auc",
        )
        best_pipe = search.best_estimator_
        res = evaluate_model(best_pipe, X_train, y_train, X_test, y_test, name=f"{name} (Tuned)")
        res["best_params"] = search.best_params_
        res["cv_best_score"] = search.best_score_
        tuned_results.append(res)
        fitted_tuned[name] = best_pipe
        print(f"    {name} tuned: test_acc={res['test_accuracy']:.3f}  "
              f"auc={res['roc_auc']:.3f}  best_params={search.best_params_}")

    tuned_df = pd.DataFrame(tuned_results)
    tuned_df.to_csv("../outputs/tuned_model_comparison.csv", index=False)

    # 8. FINAL MODEL SELECTION --------------------------------------------
    all_results = pd.concat([baseline_df, tuned_df.drop(columns=["best_params", "cv_best_score"], errors="ignore")],
                             ignore_index=True)
    all_results = all_results.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    all_results.to_csv("../outputs/all_model_comparison.csv", index=False)

    final_model_name = all_results.iloc[0]["model"]
    if final_model_name in fitted_tuned:
        final_model = fitted_tuned[final_model_name.replace(" (Tuned)", "")]
    elif final_model_name.replace(" (Tuned)", "") in fitted_tuned:
        final_model = fitted_tuned[final_model_name.replace(" (Tuned)", "")]
    else:
        final_model = fitted_baselines[final_model_name]

    print(f"\n[7] FINAL MODEL SELECTED: {final_model_name}")
    final_results = evaluate_model(final_model, X_train, y_train, X_test, y_test, name=final_model_name)
    print(f"    Train accuracy: {final_results['train_accuracy']:.3f}")
    print(f"    Test accuracy:  {final_results['test_accuracy']:.3f}")
    print(f"    Precision:      {final_results['precision']:.3f}")
    print(f"    Recall:         {final_results['recall']:.3f}")
    print(f"    F1 Score:       {final_results['f1_score']:.3f}")
    print(f"    ROC AUC:        {final_results['roc_auc']:.3f}")

    joblib.dump(final_model, "../models/final_model.joblib")
    joblib.dump(overfit_pipeline, "../models/overfit_demo_model.joblib")

    with open("../outputs/final_model_summary.json", "w") as f:
        json.dump({
            "final_model_name": final_model_name,
            **{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
               for k, v in final_results.items() if k != "model"},
        }, f, indent=2)

    # 9. PLOTS --------------------------------------------------------------
    print("\n[8] Generating evaluation plots...")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(y_test, y_pred, ax=ax, title=f"Confusion Matrix — {final_model_name}")
    fig.tight_layout()
    fig.savefig("../outputs/confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_roc_curve(y_test, y_proba, ax=ax, label=final_model_name)
    fig.tight_layout()
    fig.savefig("../outputs/roc_curve.png", dpi=150)
    plt.close(fig)

    # feature importance (use underlying model step)
    try:
        model_step = final_model.named_steps["model"]
        feat_names = get_feature_names(final_model.named_steps["preprocessor"],
                                        numeric_features, categorical_features)
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_feature_importance(model_step, feat_names, top_n=15, ax=ax)
        fig.tight_layout()
        fig.savefig("../outputs/feature_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"    (Feature importance plot skipped: {e})")

    # model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = all_results.sort_values("roc_auc")
    ax.barh(plot_df["model"], plot_df["roc_auc"], color="#2E7D6B")
    ax.set_xlabel("ROC AUC (test set)")
    ax.set_title("Model Comparison — ROC AUC")
    fig.tight_layout()
    fig.savefig("../outputs/model_comparison.png", dpi=150)
    plt.close(fig)

    # overfitting gap illustration
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Unconstrained\nDecision Tree", f"{final_model_name}\n(Final, tuned)"]
    train_vals = [overfit_results["train_accuracy"], final_results["train_accuracy"]]
    test_vals = [overfit_results["test_accuracy"], final_results["test_accuracy"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, train_vals, width, label="Train Accuracy", color="#4C7BFF")
    ax.bar(x + width/2, test_vals, width, label="Test Accuracy", color="#2E7D6B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overfitting Before vs. After Mitigation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../outputs/overfitting_comparison.png", dpi=150)
    plt.close(fig)

    # business insight plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    contract_churn = df_clean.groupby("Contract")["Churn"].mean().sort_values()
    axes[0].barh(contract_churn.index, contract_churn.values, color="#D9534F")
    axes[0].set_xlabel("Churn Rate")
    axes[0].set_title("Churn Rate by Contract Type")

    tenure_bins = pd.cut(df_clean["tenure"], bins=[0, 12, 24, 48, 72],
                          labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"])
    tenure_churn = df_clean.groupby(tenure_bins)["Churn"].mean()
    axes[1].bar(tenure_churn.index.astype(str), tenure_churn.values, color="#F0AD4E")
    axes[1].set_ylabel("Churn Rate")
    axes[1].set_title("Churn Rate by Tenure Group")
    fig.tight_layout()
    fig.savefig("../outputs/business_insights.png", dpi=150)
    plt.close(fig)

    # 10. SAVE PROCESSED SPLITS for dashboard -------------------------------
    X_test_out = X_test.copy()
    X_test_out["Churn_actual"] = y_test.values
    X_test_out["Churn_predicted"] = y_pred
    X_test_out["Churn_probability"] = y_proba
    X_test_out.to_csv("../outputs/test_predictions.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f} seconds.")
    print("Artifacts saved to ../outputs/ and ../models/")


if __name__ == "__main__":
    main()
