# deploy/app.py
r"""
Loan Eligibility Checker — Clean Final
- Human-friendly labels in UI, mapped internally to numeric codes.
- No debug outputs (no mappings, no raw features).
- Shows only: Decision, Probability, SHAP explanation.
- Includes example presets (Approved / Declined / Borderline).
"""

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
BASE_DIR = Path(r"C:\Users\meghr\OneDrive\Desktop\Loan Eligibiliy")
MODEL_PATH = BASE_DIR / "models" / "xgboost_best.joblib"
FEATURE_LIST_PATH = BASE_DIR / "artifacts" / "feature_list.txt"
DECISION_THRESHOLD = 0.5

# ---------------- Human label mappings ----------------
CATEGORY_LABELS = {
    "loan_type_enc": {
        "Conventional": 1,
        "FHA": 3,
    },
    "loan_purpose_enc": {
        "Home purchase": 1,
        "Refinance": 2,
        "Home improvement": 4,
        "Cash-out refinance": 31,
        "Other purpose": 32,
    },
    "occupancy_type_enc": {
        "Principal residence": 1,
        "Second residence": 2,
        "Investor": 3,
    },
    "property_type_enc": {
        "One-to-four family": 1,
        "Manufactured housing": 2,
        "Multifamily": 3,
    },
}

# ---------------- Loaders ----------------
@st.cache_resource(show_spinner=False)
def load_feature_list():
    if FEATURE_LIST_PATH.exists():
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    return []

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def build_shap_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

# ---------------- Helpers ----------------
def prepare_model_input(feature_list, input_values):
    row = {}
    for feat in feature_list:
        if feat == "approved_flag":
            continue
        row[feat] = input_values.get(feat, 0)
    return pd.DataFrame([row])

def align_and_select_features(model, df_row, feature_list):
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = [f for f in feature_list if f != "approved_flag"]

    df_row = df_row[[c for c in df_row.columns if c in expected]]
    for feat in expected:
        if feat not in df_row.columns:
            df_row[feat] = 0
    return df_row[expected]

def explain_prediction_text(explainer, X_row, prob, threshold=0.5, top_k=3):
    try:
        shap_values = explainer(X_row)
        values = np.array(shap_values.values).reshape(-1)
        feat_names = X_row.columns.tolist()
        order = np.argsort(-np.abs(values))
        pos = [(feat_names[i], values[i]) for i in order if values[i] > 0][:top_k]
        neg = [(feat_names[i], values[i]) for i in order if values[i] < 0][:top_k]
    except Exception:
        return f"Decision: {'Approved' if prob >= threshold else 'Rejected'} (p={prob:.3f}). No explanation."

    decision = "Approved" if prob >= threshold else "Rejected"
    lines = [f"**Decision:** {decision} (p={prob:.3f})"]
    if pos:
        lines.append("Positive contributors:")
        for fn, val in pos:
            lines.append(f"- {fn}: +{val:.3f}")
    if neg:
        lines.append("Negative contributors:")
        for fn, val in neg:
            lines.append(f"- {fn}: {val:.3f}")
    return "\n".join(lines)

def get_probability_from_model(model, X_row):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_row)[0, 1])
    pred = model.predict(X_row)
    return float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)

# ---------------- Example presets ----------------
EXAMPLES = {
    "Likely Approved": {
        "loan_amount": 100000,
        "income": 80000,
        "income_to_loan_ratio": 0.80,
        "applicant_age_num": 40,
        "loan_type_enc": "Conventional",
        "loan_purpose_enc": "Home purchase",
        "occupancy_type_enc": "Principal residence",
        "property_type_enc": "One-to-four family",
    },
    "Likely Declined": {
        "loan_amount": 400000,
        "income": 50000,
        "income_to_loan_ratio": 0.125,
        "applicant_age_num": 25,
        "loan_type_enc": "FHA",
        "loan_purpose_enc": "Cash-out refinance",
        "occupancy_type_enc": "Investor",
        "property_type_enc": "Multifamily",
    },
    "Borderline": {
        "loan_amount": 200000,
        "income": 60000,
        "income_to_loan_ratio": 0.30,
        "applicant_age_num": 30,
        "loan_type_enc": "Conventional",
        "loan_purpose_enc": "Refinance",
        "occupancy_type_enc": "Principal residence",
        "property_type_enc": "One-to-four family",
    },
}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")
st.title("Loan Eligibility Checker")
st.write("Neutral inputs only — no race, ethnicity, or location are used.")

feature_list = load_feature_list()
model = load_model()
explainer = build_shap_explainer(model)

# Example selector
example_choice = st.selectbox("Try example preset:", ["(none)"] + list(EXAMPLES.keys()))

with st.form("elig_form"):
    st.subheader("Applicant & Loan details")
    c1, c2 = st.columns(2)
    with c1:
        loan_amount = st.number_input("Loan amount (USD)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
        income = st.number_input("Applicant annual income (USD)", min_value=0.0, value=60000.0, step=500.0, format="%.2f")
        default_ratio = round(income / loan_amount, 3) if loan_amount > 0 else 0.0
        income_to_loan_ratio = st.number_input("Income to loan ratio", min_value=0.0, value=default_ratio, step=0.01, format="%.3f")
        applicant_age_num = st.number_input("Applicant age (years)", min_value=18, max_value=100, value=35, step=1)
    with c2:
        loan_type_choice = st.selectbox("Loan type", list(CATEGORY_LABELS["loan_type_enc"].keys()))
        loan_purpose_choice = st.selectbox("Loan purpose", list(CATEGORY_LABELS["loan_purpose_enc"].keys()))
        occupancy_choice = st.selectbox("Occupancy type", list(CATEGORY_LABELS["occupancy_type_enc"].keys()))
        property_choice = st.selectbox("Property type", list(CATEGORY_LABELS["property_type_enc"].keys()))

    submitted = st.form_submit_button("Check Eligibility")

# Load example if selected
if example_choice != "(none)":
    ex = EXAMPLES[example_choice]
    loan_amount = ex["loan_amount"]
    income = ex["income"]
    income_to_loan_ratio = ex["income_to_loan_ratio"]
    applicant_age_num = ex["applicant_age_num"]
    loan_type_choice = ex["loan_type_enc"]
    loan_purpose_choice = ex["loan_purpose_enc"]
    occupancy_choice = ex["occupancy_type_enc"]
    property_choice = ex["property_type_enc"]

if submitted or example_choice != "(none)":
    input_values = {
        "loan_amount": float(loan_amount),
        "income": float(income),
        "income_to_loan_ratio": float(income_to_loan_ratio),
        "applicant_age_num": int(applicant_age_num),
        "loan_type_enc": CATEGORY_LABELS["loan_type_enc"][loan_type_choice],
        "loan_purpose_enc": CATEGORY_LABELS["loan_purpose_enc"][loan_purpose_choice],
        "occupancy_type_enc": CATEGORY_LABELS["occupancy_type_enc"][occupancy_choice],
        "property_type_enc": CATEGORY_LABELS["property_type_enc"][property_choice],
    }

    X_row_init = prepare_model_input(feature_list, input_values)
    if "approved_flag" in X_row_init.columns:
        X_row_init = X_row_init.drop(columns=["approved_flag"])
    X_row = align_and_select_features(model, X_row_init, feature_list)

    try:
        prob = get_probability_from_model(model, X_row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    decision = "Approved" if prob >= DECISION_THRESHOLD else "Rejected"
    st.markdown(f"## {decision}")
    st.write(f"Predicted approval probability: **{prob:.3f}**")

    explanation_text = explain_prediction_text(explainer, X_row, prob, threshold=DECISION_THRESHOLD, top_k=3)
    st.markdown("### Explanation")
    st.markdown(explanation_text)
