
# ============================================================
# LOAN APPROVAL INTELLIGENCE
# AI-Powered Data Analytics and Prediction System
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "loan_approval_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOAN APPROVAL INTELLIGENCE")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Clean column names
df.columns = df.columns.str.strip()

# Clean categorical values
categorical_columns = [
    "education",
    "self_employed",
    "loan_status"
]

for col in categorical_columns:
    df[col] = df[col].str.strip()

# Check missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum().sum())

print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Total assets
df["total_assets"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)

# Loan-to-income ratio
df["loan_to_income_ratio"] = (
    df["loan_amount"] / df["income_annum"]
)

print("\nFeature engineering completed.")


# ============================================================
# 5. BASIC BUSINESS KPIs
# ============================================================

total_applications = len(df)

approved_count = (df["loan_status"] == "Approved").sum()
rejected_count = (df["loan_status"] == "Rejected").sum()

approval_rate = approved_count / total_applications * 100
rejection_rate = rejected_count / total_applications * 100

print("\n" + "=" * 60)
print("BUSINESS KPIs")
print("=" * 60)

print(f"Total Applications : {total_applications:,}")
print(f"Approved           : {approved_count:,}")
print(f"Rejected           : {rejected_count:,}")
print(f"Approval Rate      : {approval_rate:.2f}%")
print(f"Rejection Rate     : {rejection_rate:.2f}%")
print(f"Average Income     : ₹{df['income_annum'].mean():,.2f}")
print(f"Average Loan       : ₹{df['loan_amount'].mean():,.2f}")
print(f"Average CIBIL      : {df['cibil_score'].mean():.2f}")


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 6.1 Loan Approval Distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

status_counts = df["loan_status"].value_counts()

sns.barplot(
    x=status_counts.index,
    y=status_counts.values
)

plt.title("Loan Application Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loan_status_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.2 CIBIL Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="loan_status",
    y="cibil_score",
    data=df
)

plt.title("CIBIL Score Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("CIBIL Score")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/cibil_by_loan_status.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.3 Income Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="loan_status",
    y="income_annum",
    data=df
)

plt.title("Annual Income Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/income_by_loan_status.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.4 Loan Amount Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="loan_status",
    y="loan_amount",
    data=df
)

plt.title("Loan Amount Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loan_amount_by_status.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.5 Loan Term Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    x="loan_term",
    hue="loan_status",
    data=df
)

plt.title("Loan Term Distribution by Loan Status")
plt.xlabel("Loan Term (Years)")
plt.ylabel("Number of Applications")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loan_term_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.6 Education
# ------------------------------------------------------------

education_approval = pd.crosstab(
    df["education"],
    df["loan_status"],
    normalize="index"
) * 100

education_approval.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Loan Approval Rate by Education")
plt.xlabel("Education")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/approval_by_education.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.7 Self Employment
# ------------------------------------------------------------

employment_approval = pd.crosstab(
    df["self_employed"],
    df["loan_status"],
    normalize="index"
) * 100

employment_approval.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Loan Approval Rate by Employment Status")
plt.xlabel("Self-Employed")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/approval_by_employment.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.8 Dependents
# ------------------------------------------------------------

dependent_approval = pd.crosstab(
    df["no_of_dependents"],
    df["loan_status"],
    normalize="index"
) * 100

dependent_approval.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Loan Approval Rate by Number of Dependents")
plt.xlabel("Number of Dependents")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/approval_by_dependents.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.9 Total Assets
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="loan_status",
    y="total_assets",
    data=df
)

plt.title("Total Assets Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Total Assets")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/assets_by_status.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6.10 Loan-to-Income Ratio
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="loan_status",
    y="loan_to_income_ratio",
    data=df
)

plt.title("Loan-to-Income Ratio by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan-to-Income Ratio")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/loan_to_income_ratio.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. MACHINE LEARNING PREPARATION
# ============================================================

ml_df = df.copy()

# Target encoding
ml_df["loan_status"] = ml_df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Remove target and ID
X = ml_df.drop(
    columns=["loan_status", "loan_id"]
)

y = ml_df["loan_status"]

categorical_features = [
    "education",
    "self_employed"
]

numerical_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income_ratio"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. MODEL TRAINING
# ============================================================

models = {

    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000)
            )
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=10
                )
            )
        ]
    )
}


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

results = []
trained_models = {}
predictions = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = model
    predictions[model_name] = y_pred


# Model comparison
model_comparison = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(model_comparison.round(4))

model_comparison.round(4).to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)


# ============================================================
# 11. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_model = trained_models["Random Forest"]
rf_predictions = predictions["Random Forest"]

cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_names = (
    rf_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model
        .named_steps["classifier"]
        .feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 10 RANDOM FOREST FEATURES")
print("=" * 60)

print(rf_importance.head(10).to_string(index=False))

rf_importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)


# Feature importance chart
top_features = (
    rf_importance
    .head(10)
    .sort_values("Importance")
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top 10 Features Driving Random Forest Predictions"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/random_forest_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. 5-FOLD CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n" + "=" * 60)
print("RANDOM FOREST CROSS-VALIDATION")
print("=" * 60)

print("Fold accuracies:")
print(cv_scores)

print(
    f"\nMean Accuracy: {cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: {cv_scores.std():.4f}"
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

rf_results = model_comparison[
    model_comparison["Model"] == "Random Forest"
].iloc[0]

print(
    f"Random Forest Test Accuracy : "
    f"{rf_results['Accuracy']:.2%}"
)

print(
    f"Random Forest ROC-AUC       : "
    f"{rf_results['ROC-AUC']:.2%}"
)

print(
    f"5-Fold CV Accuracy          : "
    f"{cv_scores.mean():.2%}"
)

print(
    f"5-Fold CV Std. Deviation    : "
    f"{cv_scores.std():.4f}"
)

print(
    "\nTop predictive feature: "
    f"{rf_importance.iloc[0]['Feature']}"
)

print("\nProject execution completed successfully.")
