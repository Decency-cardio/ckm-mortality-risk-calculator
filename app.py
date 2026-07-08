from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "risk_tools" / "locked_hgb_death_model_portable.json"
SCHEMA_PATH = APP_DIR / "risk_tools" / "risk_tool_feature_schema.json"

SEX_OPTIONS = {
    "Male": 0,
    "Female": 1,
}
EDUCATION_OPTIONS = {
    "Low: below high school / primary or junior-secondary equivalent": 1,
    "Middle: high school / technical secondary / vocational equivalent": 2,
    "High: college, university, or above": 3,
}
MARITAL_OPTIONS = {
    "Married or living with spouse": 1,
    "Married, spouse absent": 2,
    "Partnered / cohabiting": 3,
    "Separated": 4,
    "Divorced": 5,
    "Separated or divorced": 6,
    "Widowed": 7,
    "Never married": 8,
}
STATE_OPTIONS = {
    "State 1: metabolic CKM without functional limitation": 1,
    "State 2: clinical cardio-kidney CKM without functional limitation": 2,
    "State 3: functional limitation": 3,
}


@st.cache_resource
def load_model_and_schema():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return model, schema


def risk_group(probability: float, low_cutoff: float, high_cutoff: float) -> str:
    if probability < low_cutoff:
        return "Low risk"
    if probability < high_cutoff:
        return "Intermediate risk"
    return "High risk"


def build_manual_row() -> dict[str, float | int]:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=45.0, max_value=105.0, value=70.0, step=1.0)
        sex_label = st.selectbox("Sex", options=list(SEX_OPTIONS.keys()))
        bmi = st.number_input("BMI, kg/m2", min_value=10.0, max_value=60.0, value=25.0, step=0.5)
        waist_cm = st.number_input("Waist circumference, cm", min_value=40.0, max_value=180.0, value=90.0, step=1.0)
        depression_score = st.number_input("Depression score", min_value=0.0, max_value=60.0, value=8.0, step=1.0)
        adl_count = st.number_input("ADL difficulty count", min_value=0.0, max_value=6.0, value=0.0, step=1.0)
        mobility_count = st.number_input("Mobility difficulty count", min_value=0.0, max_value=6.0, value=0.0, step=1.0)
        interval_waves = st.number_input(
            "Expected interval until the next follow-up visit",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=1.0,
            help="Use 1 for the usual next follow-up. This is a survey-interval indicator, not an exact number of years.",
        )
    with col2:
        from_wave = st.number_input("Current study wave", min_value=1.0, max_value=20.0, value=1.0, step=1.0)
        state_label = st.selectbox("Current CKM-functional state", options=list(STATE_OPTIONS.keys()))
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        heart_disease = st.checkbox("Heart disease")
        stroke = st.checkbox("Stroke")
        kidney_disease = st.checkbox("Kidney disease")
        current_smoking = st.checkbox("Current smoking")
        ever_smoking = st.checkbox("Ever smoking")
        current_drinking = st.checkbox("Current drinking")
        ever_drinking = st.checkbox("Ever drinking")
        education_label = st.selectbox("Education level", options=list(EDUCATION_OPTIONS.keys()), index=1)
        marital_label = st.selectbox("Marital status", options=list(MARITAL_OPTIONS.keys()), index=0)

    return {
        "age": age,
        "bmi": bmi,
        "waist_cm": waist_cm,
        "depression_score": depression_score,
        "adl_count": adl_count,
        "mobility_count": mobility_count,
        "from_wave": from_wave,
        "interval_waves": interval_waves,
        "sex": SEX_OPTIONS[sex_label],
        "education": EDUCATION_OPTIONS[education_label],
        "marital": MARITAL_OPTIONS[marital_label],
        "hypertension": int(hypertension),
        "diabetes": int(diabetes),
        "heart_disease": int(heart_disease),
        "stroke": int(stroke),
        "kidney_disease": int(kidney_disease),
        "current_smoking": int(current_smoking),
        "ever_smoking": int(ever_smoking),
        "current_drinking": int(current_drinking),
        "ever_drinking": int(ever_drinking),
        "ckm_metabolic": int(hypertension or diabetes or bmi >= 25),
        "advanced_ckm": int(heart_disease or stroke or kidney_disease),
        "from_state": STATE_OPTIONS[state_label],
    }


def _is_missing(value) -> bool:
    return bool(pd.isna(value))


def _safe_float(value, fallback: float) -> float:
    if _is_missing(value):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _transform_row(row: pd.Series, model: dict) -> list[float]:
    numeric_values: list[float] = []
    raw_numeric: list[float] = []

    for feature, statistic in zip(model["numeric_features"], model["numeric_imputer_statistics"]):
        raw_value = row.get(feature)
        missing = _is_missing(raw_value)
        raw_numeric.append(math.nan if missing else _safe_float(raw_value, float(statistic)))
        numeric_values.append(float(statistic) if missing else _safe_float(raw_value, float(statistic)))

    for feature_index in model["numeric_missing_indicator_features"]:
        numeric_values.append(1.0 if math.isnan(raw_numeric[int(feature_index)]) else 0.0)

    scaled_numeric = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(numeric_values, model["scaler_mean"], model["scaler_scale"])
    ]

    categorical_values: list[float] = []
    for feature, statistic in zip(model["categorical_features"], model["categorical_imputer_statistics"]):
        raw_value = row.get(feature)
        categorical_values.append(float(statistic) if _is_missing(raw_value) else _safe_float(raw_value, float(statistic)))

    onehot_values: list[float] = []
    for value, categories in zip(categorical_values, model["onehot_categories"]):
        onehot_values.extend(1.0 if value == float(category) else 0.0 for category in categories)

    return scaled_numeric + onehot_values


def _predict_one_probability(row: pd.Series, model: dict) -> float:
    transformed = _transform_row(row, model)
    raw_score = float(model["baseline_prediction"])

    for tree in model["trees"]:
        node_index = 0
        while True:
            node = tree[node_index]
            if int(node["is_leaf"]) == 1:
                raw_score += float(node["value"])
                break

            feature_value = transformed[int(node["feature_idx"])]
            if math.isnan(feature_value):
                node_index = int(node["left"] if int(node["missing_go_to_left"]) else node["right"])
            elif feature_value <= float(node["num_threshold"]):
                node_index = int(node["left"])
            else:
                node_index = int(node["right"])

    return 1.0 / (1.0 + math.exp(-raw_score))


def portable_predict_proba(df: pd.DataFrame, model: dict) -> list[float]:
    return [_predict_one_probability(row, model) for _, row in df.iterrows()]


def score_dataframe(df: pd.DataFrame, model: dict, schema: dict) -> pd.DataFrame:
    features = schema["numeric_features"] + schema["categorical_features"]
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    low_cutoff = float(schema["charls_fixed_risk_cutoffs"]["low_intermediate"])
    high_cutoff = float(schema["charls_fixed_risk_cutoffs"]["intermediate_high"])
    probabilities = portable_predict_proba(df[features], model)
    out = df.copy()
    out["predicted_mortality_risk_before_next_followup"] = probabilities
    out["predicted_mortality_risk_before_next_followup_pct"] = [probability * 100 for probability in probabilities]
    out["risk_group"] = [risk_group(float(p), low_cutoff, high_cutoff) for p in probabilities]
    return out


def main() -> None:
    st.set_page_config(page_title="CKM Mortality Risk Calculator", layout="centered")
    model, schema = load_model_and_schema()
    low_cutoff = float(schema["charls_fixed_risk_cutoffs"]["low_intermediate"])
    high_cutoff = float(schema["charls_fixed_risk_cutoffs"]["intermediate_high"])

    st.title("CKM Mortality Risk Before the Next Follow-Up")
    st.caption(
        "This research tool estimates the risk of being recorded as deceased by the next study follow-up. "
        "It is not a fixed 1-year or 2-year mortality risk."
    )

    tab_manual, tab_batch, tab_about = st.tabs(["Single person", "Batch CSV", "Model information"])

    with tab_manual:
        with st.form("manual_form"):
            row = build_manual_row()
            submitted = st.form_submit_button("Calculate risk")
        if submitted:
            scored = score_dataframe(pd.DataFrame([row]), model, schema)
            prob = float(scored["predicted_mortality_risk_before_next_followup"].iloc[0])
            group = str(scored["risk_group"].iloc[0])
            st.metric("Predicted mortality risk before the next follow-up", f"{prob * 100:.1f}%")
            st.metric("Risk group", group)
            st.caption(
                f"Risk groups use CHARLS-derived cutoffs: low < {low_cutoff * 100:.2f}%, "
                f"intermediate {low_cutoff * 100:.2f}% to < {high_cutoff * 100:.2f}%, "
                f"high >= {high_cutoff * 100:.2f}%."
            )

    with tab_batch:
        uploaded = st.file_uploader("Upload a CSV with the required model variables", type=["csv"])
        if uploaded is not None:
            try:
                batch = pd.read_csv(uploaded)
                scored = score_dataframe(batch, model, schema)
                st.dataframe(scored.head(20), use_container_width=True)
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ckm_mortality_risk_scored.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))

    with tab_about:
        st.write("Development cohort: CHARLS. External validation cohorts: HRS, ELSA, SHARE, and KLoSA.")
        st.write("Primary model: locked histogram gradient boosting model.")
        st.write("Prediction horizon: current assessment to the next study follow-up.")
        st.write("Risk thresholds were derived in CHARLS and applied unchanged to external cohorts.")
        st.json(schema)


if __name__ == "__main__":
    main()
