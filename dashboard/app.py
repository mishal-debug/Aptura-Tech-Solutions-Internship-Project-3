"""
app.py — Interactive Streamlit Dashboard
Aptura Tech Solutions | Telco Customer Churn Predictive Analytics Project

Run with:  streamlit run app.py
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Predictive Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():

    df_raw = pd.read_csv(DATA_DIR / "telco.csv")
    df_processed = pd.read_csv(DATA_DIR / "telco_processed.csv")
    test_preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    all_results = pd.read_csv(OUTPUT_DIR / "all_model_comparison.csv")

    with open(OUTPUT_DIR / "final_model_summary.json", "r") as f:
        final_summary = json.load(f)

    return (
        df_raw,
        df_processed,
        test_preds,
        all_results,
        final_summary,
    )


df_raw, df_processed, test_preds, all_results, final_summary = load_data()
# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Telco Customer Churn Predictive Analytics Dashboard ")
st.caption("Data Science Intern Mishal Sadiq")

st.markdown("---")

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{df_raw.shape[0]:,}")
col2.metric("Overall Churn Rate", f"{df_processed['Churn'].mean():.1%}")
col3.metric("Final Model", final_summary["final_model_name"])
col4.metric("Test Accuracy", f"{final_summary['test_accuracy']:.1%}")
col5.metric("Test ROC-AUC", f"{final_summary['roc_auc']:.3f}")

st.markdown("---")

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df_raw["Contract"].unique().tolist(),
    default=df_raw["Contract"].unique().tolist(),
)
internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df_raw["InternetService"].unique().tolist(),
    default=df_raw["InternetService"].unique().tolist(),
)
tenure_range = st.sidebar.slider(
    "Tenure (months)",
    int(df_raw["tenure"].min()),
    int(df_raw["tenure"].max()),
    (int(df_raw["tenure"].min()), int(df_raw["tenure"].max())),
)

filtered = df_raw[
    df_raw["Contract"].isin(contract_filter)
    & df_raw["InternetService"].isin(internet_filter)
    & df_raw["tenure"].between(*tenure_range)
]
st.sidebar.markdown(f"**Filtered rows:** {len(filtered):,} / {len(df_raw):,}")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Business Insights",
        "🤖 Model Comparison",
        "🎯 Model Performance",
        "🔮 Live Churn Predictor",
    ]
)

# ===============================TAB 1: Business Insights ================================
with tab1:
    st.subheader("Churn Drivers")

    c1, c2 = st.columns(2)
    with c1:
        churn_by_contract = (
            filtered.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values()
            .reset_index()
        )
        churn_by_contract.columns = ["Contract", "ChurnRate"]
        fig = px.bar(
            churn_by_contract,
            x="ChurnRate",
            y="Contract",
            orientation="h",
            color="ChurnRate",
            color_continuous_scale="Reds",
            title="Churn Rate by Contract Type",
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tenure_bins = pd.cut(
            filtered["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        )
        churn_by_tenure = (
            filtered.groupby(tenure_bins)["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index()
        )
        churn_by_tenure.columns = ["TenureGroup", "ChurnRate"]
        fig = px.bar(
            churn_by_tenure,
            x="TenureGroup",
            y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale="Oranges",
            title="Churn Rate by Tenure Group",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        churn_by_internet = (
            filtered.groupby("InternetService")["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values()
            .reset_index()
        )
        churn_by_internet.columns = ["InternetService", "ChurnRate"]
        fig = px.bar(
            churn_by_internet,
            x="InternetService",
            y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale="Purples",
            title="Churn Rate by Internet Service Type",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        churn_by_payment = (
            filtered.groupby("PaymentMethod")["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values()
            .reset_index()
        )
        churn_by_payment.columns = ["PaymentMethod", "ChurnRate"]
        fig = px.bar(
            churn_by_payment,
            x="ChurnRate",
            y="PaymentMethod",
            orientation="h",
            color="ChurnRate",
            color_continuous_scale="Blues",
            title="Churn Rate by Payment Method",
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Recommendations:** (1) Incentivize migration from month-to-month to annual "
        "contracts. (2) Launch a first-90-day onboarding program targeting new customers. "
        "(3) Bundle security/support add-ons for fiber customers. (4) Encourage autopay "
        "enrollment over electronic check."
    )

# ================================= TAB 2: Model Comparison ====================================
with tab2:
    st.subheader("Model Comparison Baseline & Tuned")
    st.dataframe(
        all_results.style.format(
            {
                "train_accuracy": "{:.3f}",
                "test_accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1_score": "{:.3f}",
                "roc_auc": "{:.3f}",
                "train_test_gap": "{:.3f}",
            }
        ).background_gradient(subset=["roc_auc"], cmap="Greens"),
        use_container_width=True,
    )

    fig = px.bar(
        all_results.sort_values("roc_auc"),
        x="roc_auc",
        y="model",
        orientation="h",
        color="roc_auc",
        color_continuous_scale="Teal",
        title="ROC-AUC by Model",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            name="Train Accuracy",
            x=all_results["model"],
            y=all_results["train_accuracy"],
        )
    )
    fig2.add_trace(
        go.Bar(
            name="Test Accuracy", x=all_results["model"], y=all_results["test_accuracy"]
        )
    )
    fig2.update_layout(
        barmode="group",
        title="Train vs Test Accuracy Overfitting Check",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Models with a large gap between train and test accuracy (e.g., an unregularized "
        "Decision Tree or Random Forest) are overfitting. The selected final model keeps "
        "this gap small while maximizing test ROC-AUC."
    )

# ================================= TAB 3: Model Performance ====================================
with tab3:
    st.subheader(f"Final Model: {final_summary['final_model_name']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{final_summary['precision']:.1%}")
    m2.metric("Recall", f"{final_summary['recall']:.1%}")
    m3.metric("F1 Score", f"{final_summary['f1_score']:.1%}")
    m4.metric("Train/Test Gap", f"{final_summary['train_test_gap']:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(test_preds["Churn_actual"], test_preds["Churn_predicted"])
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        from sklearn.metrics import roc_auc_score, roc_curve

        fpr, tpr, _ = roc_curve(
            test_preds["Churn_actual"], test_preds["Churn_probability"]
        )
        auc = roc_auc_score(test_preds["Churn_actual"], test_preds["Churn_probability"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})")
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest Churn-Risk Customers (Test Set)")
    risky = test_preds.sort_values("Churn_probability", ascending=False).head(15)
    show_cols = [
        "tenure",
        "Contract",
        "MonthlyCharges",
        "InternetService",
        "Churn_actual",
        "Churn_predicted",
        "Churn_probability",
    ]
    show_cols = [c for c in show_cols if c in risky.columns]
    st.dataframe(
        risky[show_cols].style.format({"Churn_probability": "{:.1%}"}),
        use_container_width=True,
    )

# ================================= TAB 4: Live Predictor ====================================
with tab4:
    st.subheader("🔮 Score a Hypothetical Customer")
    st.caption(
        "Adjust the inputs and get a live churn-probability prediction from the trained pipeline."
    )

    model = joblib.load(MODEL_DIR / "final_model.joblib")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with c2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
    with c3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)

    total_charges = monthly_charges * tenure

    if st.button("Predict Churn Risk", type="primary"):
        service_cols_vals = [
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]
        num_services = sum(
            1
            for v in service_cols_vals
            if v not in ("No", "No internet service", "No phone service")
        )

        row = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "tenure_years": tenure / 12.0,
                    "avg_monthly_spend": (
                        (total_charges / tenure) if tenure > 0 else monthly_charges
                    ),
                    "num_services": num_services,
                    "is_new_customer": int(tenure <= 3),
                    "charge_per_service": monthly_charges / max(num_services, 1),
                }
            ]
        )

        proba = model.predict_proba(row)[0, 1]
        pred = "CHURN" if proba >= 0.5 else "RETAIN"

        st.markdown("### Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Predicted Churn Probability", f"{proba:.1%}")
            st.markdown(f"**Prediction: {pred}**")
        with r2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Churn Risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#08080E" if proba >= 0.5 else "#2E0669"},
                        "steps": [
                            {"range": [0, 33], "color": "#A5B4A7"},
                            {"range": [33, 66], "color": "#9F9992"},
                            {"range": [66, 100], "color": "#90939B"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Built for Aptura Tech Solutions · Data Science Internship Batch 02 · Week 03 Task . By Mishal Sadiq"
)