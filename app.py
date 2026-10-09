import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Loan Approval Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e5eaf2;
        padding: 14px 16px; border-radius: 12px;
    }
    .small-note {color: #64748b; font-size: 0.88rem;}
</style>
""", unsafe_allow_html=True)

st.title("Loan Approval Intelligence")
st.caption("Loan portfolio analytics • Machine-learning model comparison • Applicant prediction")
st.info(
    "This is an educational decision-support prototype trained on historical data. "
    "Predictions are not credit decisions and must not be used as the sole basis for lending."
)

EXPECTED_COLUMNS = [
    "loan_id", "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value", "loan_status"
]
NUMERIC_BASE = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value"
]
CATEGORICAL = ["education", "self_employed"]
TARGET_LABELS = {"Approved": 1, "Rejected": 0}

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lower()
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "The uploaded CSV does not have the expected loan approval dataset columns. "
            "Missing: " + ", ".join(missing_cols)
        )
    df = df.copy()
    for col in CATEGORICAL + ["loan_status"]:
        df[col] = df[col].astype(str).str.strip()
    # Standardise common variants while preserving the source categories.
    df["loan_status"] = df["loan_status"].str.title()
    df["education"] = df["education"].str.title()
    df["self_employed"] = df["self_employed"].str.title()
    for col in NUMERIC_BASE + ["loan_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_BASE + CATEGORICAL + ["loan_status"])
    df = df[df["loan_status"].isin(["Approved", "Rejected"])]
    if df.empty:
        raise ValueError("No usable rows remained after cleaning. Check the CSV values.")
    df["total_assets"] = (
        df["residential_assets_value"] + df["commercial_assets_value"]
        + df["luxury_assets_value"] + df["bank_asset_value"]
    )
    df["loan_to_income_ratio"] = np.where(
        df["income_annum"] > 0,
        df["loan_amount"] / df["income_annum"],
        np.nan
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["loan_to_income_ratio"])
    return df

@st.cache_resource(show_spinner="Training the three models on your dataset…")
def train_models(data_bytes, test_size, random_state):
    df = load_and_clean(data_bytes)
    features = NUMERIC_BASE + ["total_assets", "loan_to_income_ratio"] + CATEGORICAL
    X = df[features].copy()
    y = df["loan_status"].map(TARGET_LABELS).astype(int)
    if y.nunique() < 2:
        raise ValueError("The dataset must contain both Approved and Rejected applications.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"),
             CATEGORICAL)
        ],
        remainder="passthrough"
    )
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=random_state, max_depth=5))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=random_state, max_depth=10
            ))
        ])
    }
    fitted, metric_rows, predictions, probabilities = {}, [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        fitted[name] = model
        predictions[name] = pred
        probabilities[name] = prob
        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
    return {
        "df": df, "X_test": X_test, "y_test": y_test,
        "models": fitted, "predictions": predictions,
        "probabilities": probabilities, "metrics": pd.DataFrame(metric_rows),
        "features": features, "test_size": test_size, "random_state": random_state
    }

with st.sidebar:
    st.header("Data & settings")
    uploaded = st.file_uploader(
        "Upload a different loan dataset (optional)",
        type=["csv"],
        help="If no file is uploaded, the app loads loan_approval_dataset.csv from the repository when available."
    )
    if os.path.exists("loan_approval_dataset.csv"):
        st.caption("Default project dataset is available. Upload another compatible CSV only if you want to replace it for this session.")
    else:
        st.caption("No default CSV found in the app files. Upload loan_approval_dataset.csv to populate the dashboard.")
    test_size_pct = st.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    st.markdown("---")
    st.caption("Models: Logistic Regression, Decision Tree, Random Forest")

if uploaded is not None:
    data_bytes = uploaded.getvalue()
    data_source_label = "uploaded CSV"
elif os.path.exists("loan_approval_dataset.csv"):
    with open("loan_approval_dataset.csv", "rb") as default_file:
        data_bytes = default_file.read()
    data_source_label = "default project dataset"
else:
    st.markdown("### Get started")
    st.write(
        "To populate the dashboard automatically, add `loan_approval_dataset.csv` "
        "to the root of your GitHub repository alongside `app.py`. Until then, upload "
        "the CSV using the sidebar."
    )
    st.markdown("**Expected columns**")
    st.code(", ".join(EXPECTED_COLUMNS), language="text")
    st.markdown(
        "Use the public dataset source linked in your "
        "[GitHub README](https://github.com/pranjaligupta1st/loan-approval-intelligence). "
        "Do not commit private or sensitive applicant data."
    )
    st.stop()

try:
    df = load_and_clean(data_bytes)
    result = train_models(data_bytes, test_size_pct / 100, int(seed))
    st.sidebar.success(f"Data source: {data_source_label}")
except Exception as exc:
    st.error(f"Could not prepare the dashboard: {exc}")
    st.stop()

df = result["df"]
metrics = result["metrics"]
best_row = metrics.sort_values("F1 Score", ascending=False).iloc[0]
approved = int((df["loan_status"] == "Approved").sum())
rejected = int((df["loan_status"] == "Rejected").sum())
approval_rate = approved / len(df) * 100

tab_overview, tab_analytics, tab_prediction, tab_models, tab_data = st.tabs([
    "Overview", "Portfolio Analytics", "Loan Prediction", "Model Performance", "Dataset"
])

with tab_overview:
    st.subheader("Portfolio overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(df):,}")
    c2.metric("Approved", f"{approved:,}")
    c3.metric("Approval rate", f"{approval_rate:.1f}%")
    c4.metric("Average CIBIL score", f"{df['cibil_score'].mean():,.0f}")
    c5, c6, c7 = st.columns(3)
    c5.metric("Rejected", f"{rejected:,}")
    c6.metric("Average annual income", f"₹{df['income_annum'].mean():,.0f}")
    c7.metric("Average loan amount", f"₹{df['loan_amount'].mean():,.0f}")
    left, right = st.columns(2)
    with left:
        status_df = df["loan_status"].value_counts().rename_axis("Status").reset_index(name="Applications")
        fig = px.bar(status_df, x="Status", y="Applications", color="Status",
                     title="Application outcomes", text="Applications")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(df, x="cibil_score", color="loan_status", barmode="overlay",
                           opacity=0.7, title="CIBIL score distribution by outcome")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Current model selection by test-set F1 score: {best_row['Model']} "
        f"(F1 = {best_row['F1 Score']:.3f}). This is calculated from the uploaded dataset "
        "and selected train/test split; it is not a guarantee of future performance."
    )

with tab_analytics:
    st.subheader("Explore approval patterns")
    a1, a2 = st.columns(2)
    with a1:
        group_col = st.selectbox(
            "Compare approval outcomes by",
            ["education", "self_employed", "loan_term", "no_of_dependents"],
            format_func=lambda x: {
                "education": "Education", "self_employed": "Self-employment",
                "loan_term": "Loan term (years)", "no_of_dependents": "Number of dependents"
            }[x]
        )
        grouped = pd.crosstab(df[group_col].astype(str), df["loan_status"], normalize="index") * 100
        grouped = grouped.reset_index().melt(id_vars=group_col, var_name="Outcome", value_name="Share (%)")
        fig = px.bar(grouped, x=group_col, y="Share (%)", color="Outcome", barmode="group",
                     title="Outcome share within each group")
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        numeric_col = st.selectbox(
            "Compare numeric variable",
            ["income_annum", "loan_amount", "cibil_score", "total_assets", "loan_to_income_ratio"],
            format_func=lambda x: {
                "income_annum": "Annual income", "loan_amount": "Loan amount",
                "cibil_score": "CIBIL score", "total_assets": "Total assets",
                "loan_to_income_ratio": "Loan-to-income ratio"
            }[x]
        )
        fig = px.box(df, x="loan_status", y=numeric_col, color="loan_status",
                     points="outliers", title=f"{numeric_col.replace('_', ' ').title()} by outcome")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Descriptive statistics by outcome")
    st.dataframe(
        df.groupby("loan_status")[["income_annum", "loan_amount", "cibil_score",
                                   "total_assets", "loan_to_income_ratio"]]
          .agg(["count", "mean", "median"]).round(2),
        use_container_width=True
    )
    st.caption("These are associations in the historical dataset, not proof that a variable causes approval.")

with tab_prediction:
    st.subheader("Predict an applicant's outcome")
    st.write("Enter applicant details. The selected model is trained on the uploaded dataset.")
    model_name = st.selectbox("Prediction model", list(result["models"].keys()), index=2)
    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            dependents = st.number_input("Number of dependents", 0, 20, 1)
            education = st.selectbox("Education", sorted(df["education"].unique().tolist()))
            self_employed = st.selectbox("Self-employed?", sorted(df["self_employed"].unique().tolist()))
            income = st.number_input("Annual income", min_value=1, value=int(df["income_annum"].median()), step=10000)
        with p2:
            loan_amount = st.number_input("Loan amount", min_value=1, value=int(df["loan_amount"].median()), step=10000)
            loan_term = st.number_input("Loan term (years)", min_value=1, max_value=50,
                                        value=int(max(1, round(df["loan_term"].median()))))
            cibil = st.number_input("CIBIL score", min_value=300, max_value=900,
                                    value=int(np.clip(round(df["cibil_score"].median()), 300, 900)))
            residential = st.number_input("Residential assets value", min_value=0,
                                          value=int(max(0, df["residential_assets_value"].median())), step=10000)
        with p3:
            commercial = st.number_input("Commercial assets value", min_value=0,
                                          value=int(max(0, df["commercial_assets_value"].median())), step=10000)
            luxury = st.number_input("Luxury assets value", min_value=0,
                                     value=int(max(0, df["luxury_assets_value"].median())), step=10000)
            bank_assets = st.number_input("Bank asset value", min_value=0,
                                          value=int(max(0, df["bank_asset_value"].median())), step=10000)
        submitted = st.form_submit_button("Predict loan outcome", type="primary")
    if submitted:
        applicant = pd.DataFrame([{
            "no_of_dependents": dependents,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": residential,
            "commercial_assets_value": commercial,
            "luxury_assets_value": luxury,
            "bank_asset_value": bank_assets,
            "total_assets": residential + commercial + luxury + bank_assets,
            "loan_to_income_ratio": loan_amount / income,
            "education": education,
            "self_employed": self_employed
        }])[result["features"]]
        model = result["models"][model_name]
        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]
        approved_prob = float(probabilities[list(model.classes_).index(1)])
        if prediction == 1:
            st.success("Model prediction: Likely Approved")
        else:
            st.warning("Model prediction: Likely Rejected")
        st.metric("Model-estimated approval probability", f"{approved_prob:.1%}")
        st.progress(approved_prob)
        st.caption(
            "This is a model output based on historical data, not a lender's decision or a "
            "validated estimate of an individual's real-world creditworthiness."
        )

with tab_models:
    st.subheader("Model performance on held-out test data")
    st.write(
        f"Split: {int((1-result['test_size'])*100)}% training / "
        f"{int(result['test_size']*100)}% test • random seed {result['random_state']}."
    )
    st.dataframe(metrics.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1 Score": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True)
    metric_to_plot = st.selectbox("Metric to compare", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
    fig = px.bar(metrics, x="Model", y=metric_to_plot, text_auto=".3f",
                 range_y=[0, 1], title=f"{metric_to_plot} by model")
    st.plotly_chart(fig, use_container_width=True)
    selected_model = st.selectbox("Inspect confusion matrix for", list(result["models"].keys()))
    cm = confusion_matrix(result["y_test"], result["predictions"][selected_model], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual Rejected", "Actual Approved"],
                         columns=["Predicted Rejected", "Predicted Approved"])
    st.markdown("#### Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    st.caption("Rows are actual outcomes; columns are model predictions.")
    st.markdown("#### Classification report")
    report = classification_report(
        result["y_test"], result["predictions"][selected_model],
        labels=[0, 1], target_names=["Rejected", "Approved"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    rf = result["models"]["Random Forest"]
    prep = rf.named_steps["preprocessor"]
    clf = rf.named_steps["classifier"]
    try:
        feature_names = prep.get_feature_names_out()
        importances = pd.DataFrame({"Feature": feature_names, "Importance": clf.feature_importances_})
        importances = importances.sort_values("Importance", ascending=False).head(12).sort_values("Importance")
        st.markdown("#### Random Forest feature importance")
        fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                     title="Top features used by the Random Forest model")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Feature importance describes model reliance, not causal influence or fairness.")
    except Exception as exc:
        st.warning(f"Could not display feature importance: {exc}")

with tab_data:
    st.subheader("Dataset preview")
    st.write(f"Rows after cleaning: **{len(df):,}** · Columns: **{df.shape[1]}**")
    st.dataframe(df.head(100), use_container_width=True)
    with st.expander("Data quality summary"):
        quality = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
            "Unique values": [int(df[c].nunique()) for c in df.columns]
        })
        st.dataframe(quality, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset", csv, "cleaned_loan_approval_dataset.csv", "text/csv")

st.markdown("---")
st.markdown(
    '<p class="small-note">Loan Approval Intelligence • Educational prototype • '
    'Model metrics are recalculated from the uploaded dataset on this app. '
    'No performance values are hard-coded.</p>',
    unsafe_allow_html=True
)
