# Loan Approval Intelligence

## AI-Powered Data Analytics and Prediction System for Smarter Loan Decisions

### Project Overview

Loan approval decisions depend on multiple applicant and loan characteristics such as credit score, income, loan amount, loan term, assets, education, and employment status.

This project uses data analytics and machine learning to understand the factors associated with loan approval and build predictive models that can estimate whether a loan application is likely to be approved.

The project follows a complete analytics workflow:

**Data → Information → Insights → Prediction → Decision Support**

---

## Business Problem

Financial institutions need to evaluate loan applications efficiently while managing credit risk. Analysing historical application data can help identify important patterns in approval outcomes and support more consistent decision-making.

### Key Question

**What factors influence loan approval, and can machine learning help predict whether a loan application is likely to be approved?**

---

## Objectives

- Analyse the characteristics of approved and rejected loan applications.
- Identify important patterns and relationships in the data.
- Engineer useful financial features such as total assets and loan-to-income ratio.
- Compare multiple machine learning classification models.
- Evaluate model performance using accuracy, precision, recall, F1-score and ROC-AUC.
- Identify the features that contribute most to model predictions.
- Provide data-driven insights that can support loan decision-making.

---

## Dataset

The project uses a loan approval dataset containing **4,269 loan applications and 13 original variables**.

### Main Variables

- `loan_id`
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status`

The target variable is **`loan_status`**, representing whether a loan application was approved or rejected.

---

## Data Preparation

The following preprocessing steps were performed:

- Checked for missing values.
- Checked for duplicate records.
- Cleaned column names and categorical values.
- Converted categorical variables into machine-readable features using One-Hot Encoding.
- Excluded `loan_id` from the machine learning features.

The dataset contained **no missing values and no duplicate rows**.

---

## Feature Engineering

Two additional features were created:

### Total Assets

```text
total_assets =
residential_assets_value
+ commercial_assets_value
+ luxury_assets_value
+ bank_asset_value
