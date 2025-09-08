# deploy/app.py
r"""
Loan Eligibility Checker — show text labels in UI and map to codes
- Uses manual CATEGORY_LABELS if present (edit to set real human labels).
- Falls back to artifacts/category_labels.json -> encodings.json -> cleaned_strict.csv.
- Drops target 'approved_flag' and aligns feature order to model.
"""
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
import json
import warnings
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
BASE_DIR = Path(r"C:\Users\meghr\OneDrive\Desktop\Loan Eligibiliy")
MODEL_PATH = BASE_DIR / "models" / "xgboost_best.joblib"
FEATURE_LIST_PATH = BASE_DIR / "artifacts" / "feature_list.txt"
ENCODINGS_PATH = BASE_DIR / "artifacts" / "encodings.json"
CLEANED_PATH = BASE_DIR / "artifacts" / "cleaned_strict.csv"
CATEGORY_LABELS_JSON = BASE_DIR / "artifacts" / "category_labels.json"  # optional external mapping
DECISION_THRESHOLD = 0.5

# ---------------- Manual label mappings (edit if you want permanent custom labels) ----------------
# You had these before; keep/edit as desired. UI will prefer these human labels.
CATEGORY_LABELS: Dict[str, Dict[str, Any]] = {
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

CATEGORICAL_KEYS = ["loan_type_enc", "loan_purpose_enc", "occupancy_type_enc", "property_type_enc"]

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_feature_list():
    if not FEATURE_LIST_PATH.exists():
        return []
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

@st.cache_resource(show_spinner=False)
def load_encodings():
    if not ENCODINGS_PATH.exists():
        return {}
    try:
        return json.load(open(ENCODINGS_PATH, "r", encoding="utf-8"))
    except Exception:
        return {}

@st.cache_resource(show_spinner=False)
def load_cleaned_sample(nrows: int = 2000):
    if not CLEANED_PATH.exists():
        return None
    try:
        return pd.read_csv(CLEANED_PATH, nrows=nrows, low_memory=False)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_persisted_category_labels():
    if not CATEGORY_LABELS_JSON.exists():
        return {}
    try:
        return json.load(open(CATEGORY_LABELS_JSON, "r", encoding="utf-8"))
    except Exception:
        return {}

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed loading model: {e}")
        st.stop()

@st.cache_resource(show_spinner=False)
def build_shap_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

# ---------------- Utility helpers for labels & mapping ----------------
def is_numeric_like(x) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False

def labels_from_encodings(encodings: dict, key: str) -> Dict[str, Any]:
    """
    Try to extract label->code mapping from encodings.json.
    Handles code->label or label->code shapes.
    """
    if not isinstance(encodings, dict):
        return {}
    candidates = [key, key.replace("_enc", ""), key.replace("_enc", "") + "_1", "derived_" + key.replace("_enc", "")]
    for c in candidates:
        if c in encodings and isinstance(encodings[c], dict):
            raw = encodings[c]
            keys = list(raw.keys())
            vals = list(raw.values())
            # code->label (keys numeric strings, values strings)
            if all(is_numeric_like(k) for k in keys) and any(isinstance(v, str) for v in vals):
                return {str(v): int(k) if str(k).isdigit() else k for k, v in raw.items()}
            # label->code
            if not all(is_numeric_like(k) for k in keys):
                out = {}
                for k, v in raw.items():
                    out[str(k)] = int(v) if (isinstance(v, (int, np.integer)) or (isinstance(v, str) and str(v).isdigit())) else v
                return out
    return {}

def labels_from_cleaned(cleaned_df: pd.DataFrame, field_key: str) -> Dict[str, Any]:
    """
    Infer labels by enumerating cleaned_strict.csv column values and producing labels like "Loan type 1".
    Returns map label->code.
    """
    if cleaned_df is None:
        return {}
    base = field_key.replace("_enc", "")
    candidates = [field_key, base, base + "_1", "derived_" + base]
    colname = None
    for c in candidates:
        if c in cleaned_df.columns:
            colname = c
            break
    if colname is None:
        # try substring match
        for col in cleaned_df.columns:
            if base in col.lower():
                colname = col
                break
    if colname is None:
        return {}
    vals = cleaned_df[colname].dropna().unique().tolist()
    # sort numerically if possible
    def keyfunc(x):
        try:
            return int(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return str(x)
    vals_sorted = sorted(vals, key=keyfunc)
    out = {}
    for v in vals_sorted:
        lab = f"{base.replace('_', ' ').title()} {v}"
        out[str(lab)] = int(v) if (isinstance(v, (int, np.integer)) or (isinstance(v, str) and str(v).isdigit())) else v
    return out

def get_label_mapping_for_field(field_key: str, encodings: dict, cleaned_df: pd.DataFrame, persisted: dict) -> Dict[str, Any]:
    """
    Returns label->code mapping for the field. Priority:
      1. CATEGORY_LABELS (hard-coded in file)
      2. persisted JSON (artifacts/category_labels.json)
      3. encodings.json
      4. cleaned_strict.csv inferred labels
      5. fallback to empty {}
    """
    # 1. hard-coded CATEGORY_LABELS in the file (highest priority)
    if field_key in CATEGORY_LABELS and isinstance(CATEGORY_LABELS[field_key], dict) and CATEGORY_LABELS[field_key]:
        return CATEGORY_LABELS[field_key]

    # 2. persisted JSON
    if field_key in persisted and isinstance(persisted[field_key], dict) and persisted[field_key]:
        return persisted[field_key]

    # 3. encodings.json
    e = labels_from_encodings(encodings, field_key)
    if e:
        return e

    # 4. cleaned CSV
    c = labels_from_cleaned(cleaned_df, field_key)
    if c:
        return c

    # 5. fallback empty
    return {}

def options_for_select(field_key: str, encodings: dict, cleaned_df: pd.DataFrame, persisted: dict) -> list:
    """
    Returns list of human labels to populate selectbox (preserves order from mapping where possible).
    """
    cmap = get_label_mapping_for_field(field_key, encodings, cleaned_df, persisted)
    if not cmap:
        return []
    # try preserving numeric order of codes when mapping came from cleaned or encodings numeric keys
    return list(cmap.keys())

def map_label_to_code(field_key: str, human_label: str, encodings: dict, cleaned_df: pd.DataFrame, persisted: dict):
    cmap = get_label_mapping_for_field(field_key, encodings, cleaned_df, persisted)
    if not cmap:
        # try numeric cast
        try:
            return int(float(human_label))
        except Exception:
            return human_label
    # exact match
    if human_label in cmap:
        return cmap[human_label]
    # case-insensitive
    for k in cmap:
        if k.lower() == str(human_label).lower():
            return cmap[k]
    # fallback numeric cast
    try:
        return int(float(human_label))
    except Exception:
        return human_label

# ---------------- Model input & alignment ----------------
def prepare_model_input(feature_list: list, input_values: dict) -> pd.DataFrame:
    row = {}
    for feat in feature_list:
        if feat == "approved_flag":
            continue
        # If user supplied the feature directly (like encoded names), use it
        if feat in input_values:
            row[feat] = input_values[feat]
        else:
            # try base name (e.g., model expects loan_type_enc but input given as loan_type_enc)
            row[feat] = input_values.get(feat, 0)
    # include any extra provided inputs (but we'll align later)
    for k, v in input_values.items():
        if k not in row and k != "approved_flag":
            row[k] = v
    df = pd.DataFrame([row])
    return df

def align_and_select_features(model, df_row: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = [f for f in feature_list if f != "approved_flag"]

    # drop extras
    extra = [c for c in df_row.columns if c not in expected]
    if extra:
        df_row = df_row.drop(columns=extra)

    # add missing
    for feat in expected:
        if feat not in df_row.columns:
            df_row[feat] = 0

    # reorder
    df_row = df_row[expected]

    # coerce numeric where possible
    for c in df_row.columns:
        if df_row[c].dtype == "object":
            coerced = pd.to_numeric(df_row[c], errors="coerce")
            if coerced.notna().sum() > 0:
                df_row[c] = coerced.fillna(0)
            else:
                try:
                    df_row[c] = df_row[c].astype(float)
                except Exception:
                    df_row[c] = 0
    return df_row

# ---------------- Explain / predict helpers ----------------
def build_shap_explainer_safe(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

def explain_prediction_text(model, explainer, X_row, prob, top_k=3):
    try:
        if explainer is None:
            raise RuntimeError("No SHAP")
        sh = explainer(X_row)
        vals = sh.values
        if isinstance(vals, list):
            try:
                vals = vals[1]
            except Exception:
                vals = vals[0]
        vals = np.array(vals).reshape(-1)
        feat_names = X_row.columns.tolist()
        order = np.argsort(-np.abs(vals))
        pos = [(feat_names[i], float(vals[i])) for i in order if vals[i] > 0][:top_k]
        neg = [(feat_names[i], float(vals[i])) for i in order if vals[i] < 0][:top_k]
    except Exception:
        decision = "Approved" if prob >= DECISION_THRESHOLD else "Rejected"
        if decision == "Approved":
            return f"**Decision:** Approved. (p = {prob:.3f}). Model explanation not available."
        else:
            return f"**Decision:** Rejected. (p = {prob:.3f}). Model explanation not available."

    decision = "Approved" if prob >= DECISION_THRESHOLD else "Rejected"
    lines = [f"**Decision:** {decision}. (p = {prob:.3f})"]
    if pos:
        lines.append("Top positive contributors:")
        for fn, v in pos:
            lines.append(f"• {fn}: +{v:.3f}")
    if neg:
        lines.append("Top negative contributors:")
        for fn, v in neg:
            lines.append(f"• {fn}: {v:.3f}")
    return "\n\n".join(lines)

def predict_probability(model, X_row: pd.DataFrame):
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X_row)[0, 1])
    except Exception:
        pass
    pred = model.predict(X_row)
    if isinstance(pred, (list, np.ndarray)):
        return float(pred[0])
    return float(pred)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")
st.title("Loan Eligibility Checker")
st.write("Neutral inputs only — no race/ethnicity/location are requested.")

# load artifacts/model
feature_list = load_feature_list()
encodings = load_encodings()
cleaned_df = load_cleaned_sample()
persisted_labels = load_persisted_category_labels()
model = load_model()
explainer = build_shap_explainer_safe(model)

# build UI label lists (prefer file-level CATEGORY_LABELS, then persisted, then encodings, then cleaned)
loan_type_opts = options_for_select("loan_type_enc", encodings, cleaned_df, persisted_labels)
loan_purpose_opts = options_for_select("loan_purpose_enc", encodings, cleaned_df, persisted_labels)
occupancy_opts = options_for_select("occupancy_type_enc", encodings, cleaned_df, persisted_labels)
property_opts = options_for_select("property_type_enc", encodings, cleaned_df, persisted_labels)

# If no mapping found anywhere, but user provided CATEGORY_LABELS hard-coded, prefer that
if not loan_type_opts and "loan_type_enc" in CATEGORY_LABELS:
    loan_type_opts = list(CATEGORY_LABELS["loan_type_enc"].keys())
if not loan_purpose_opts and "loan_purpose_enc" in CATEGORY_LABELS:
    loan_purpose_opts = list(CATEGORY_LABELS["loan_purpose_enc"].keys())
if not occupancy_opts and "occupancy_type_enc" in CATEGORY_LABELS:
    occupancy_opts = list(CATEGORY_LABELS["occupancy_type_enc"].keys())
if not property_opts and "property_type_enc" in CATEGORY_LABELS:
    property_opts = list(CATEGORY_LABELS["property_type_enc"].keys())

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
        if loan_type_opts:
            loan_type_choice = st.selectbox("Loan type", loan_type_opts)
        else:
            loan_type_choice = st.text_input("Loan type (enter label or code)", value="")

        if loan_purpose_opts:
            loan_purpose_choice = st.selectbox("Loan purpose", loan_purpose_opts)
        else:
            loan_purpose_choice = st.text_input("Loan purpose (enter label or code)", value="")

        if occupancy_opts:
            occupancy_choice = st.selectbox("Occupancy type", occupancy_opts)
        else:
            occupancy_choice = st.text_input("Occupancy type (enter label or code)", value="")

        if property_opts:
            property_choice = st.selectbox("Property type", property_opts)
        else:
            property_choice = st.text_input("Property type (enter label or code)", value="")

    # quick examples (optional)
    example_choice = st.selectbox("Try example:", ["(none)", "Likely Approved", "Likely Declined", "Borderline"])
    submitted = st.form_submit_button("Check Eligibility")

# load examples
if example_choice != "(none)":
    if example_choice == "Likely Approved":
        loan_amount = 100000.0; income = 80000.0; income_to_loan_ratio = 0.8; applicant_age_num = 40
        loan_type_choice = loan_type_opts[0] if loan_type_opts else list(CATEGORY_LABELS.get("loan_type_enc", {}).keys())[0]
        loan_purpose_choice = loan_purpose_opts[0] if loan_purpose_opts else list(CATEGORY_LABELS.get("loan_purpose_enc", {}).keys())[0]
        occupancy_choice = occupancy_opts[0] if occupancy_opts else list(CATEGORY_LABELS.get("occupancy_type_enc", {}).keys())[0]
        property_choice = property_opts[0] if property_opts else list(CATEGORY_LABELS.get("property_type_enc", {}).keys())[0]
    elif example_choice == "Likely Declined":
        loan_amount = 400000.0; income = 50000.0; income_to_loan_ratio = 0.125; applicant_age_num = 25
        # choose last option if available
        loan_type_choice = loan_type_opts[-1] if loan_type_opts else list(CATEGORY_LABELS.get("loan_type_enc", {}).keys())[-1]
        loan_purpose_choice = loan_purpose_opts[-1] if loan_purpose_opts else list(CATEGORY_LABELS.get("loan_purpose_enc", {}).keys())[-1]
        occupancy_choice = occupancy_opts[-1] if occupancy_opts else list(CATEGORY_LABELS.get("occupancy_type_enc", {}).keys())[-1]
        property_choice = property_opts[-1] if property_opts else list(CATEGORY_LABELS.get("property_type_enc", {}).keys())[-1]
    else:
        loan_amount = 200000.0; income = 60000.0; income_to_loan_ratio = 0.3; applicant_age_num = 30
        loan_type_choice = loan_type_opts[0] if loan_type_opts else list(CATEGORY_LABELS.get("loan_type_enc", {}).keys())[0]
        loan_purpose_choice = loan_purpose_opts[1] if len(loan_purpose_opts) > 1 else list(CATEGORY_LABELS.get("loan_purpose_enc", {}).keys())[1]
        occupancy_choice = occupancy_opts[0] if occupancy_opts else list(CATEGORY_LABELS.get("occupancy_type_enc", {}).keys())[0]
        property_choice = property_opts[0] if property_opts else list(CATEGORY_LABELS.get("property_type_enc", {}).keys())[0]

if submitted:
    # Map human labels to numeric codes before prediction
    persisted = persisted_labels
    input_values = {
        "loan_amount": float(loan_amount),
        "income": float(income),
        "income_to_loan_ratio": float(income_to_loan_ratio),
        "applicant_age_num": int(applicant_age_num),
        "loan_type_enc": map_label_to_code("loan_type_enc", loan_type_choice, encodings, cleaned_df, persisted),
        "loan_purpose_enc": map_label_to_code("loan_purpose_enc", loan_purpose_choice, encodings, cleaned_df, persisted),
        "occupancy_type_enc": map_label_to_code("occupancy_type_enc", occupancy_choice, encodings, cleaned_df, persisted),
        "property_type_enc": map_label_to_code("property_type_enc", property_choice, encodings, cleaned_df, persisted),
    }

    st.write("#### Human→code mapping used (for categorical fields)")
    st.write({
        "loan_type": input_values["loan_type_enc"],
        "loan_purpose": input_values["loan_purpose_enc"],
        "occupancy": input_values["occupancy_type_enc"],
        "property_type": input_values["property_type_enc"],
    })

    # Build model input and align features (drops approved_flag)
    X_row_init = prepare_model_input(feature_list, input_values)
    if "approved_flag" in X_row_init.columns:
        X_row_init = X_row_init.drop(columns=["approved_flag"])
    X_row = align_and_select_features(model, X_row_init, feature_list)

    st.write("#### Exact model input (feature -> value):")
    st.dataframe(X_row.T.rename(columns={0: "value"}))

    # predict
    try:
        prob = predict_probability(model, X_row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    decision = "Approved" if prob >= DECISION_THRESHOLD else "Rejected"
    st.markdown(f"## {decision}")
    st.write(f"Predicted approval probability: **{prob:.4f}**")

    # explanation
    explanation_text = explain_prediction_text(model, explainer, X_row, prob, top_k=3)
    st.markdown("### Explanation")
    st.markdown(explanation_text)

    st.info("To permanently set nicer human labels, edit the CATEGORY_LABELS dict in this file or add artifacts/category_labels.json with mappings label->code.")
