# CKM Mortality Risk Calculator

This repository contains a Streamlit web application for individualized prediction of mortality risk before the next study follow-up among individuals with cardiovascular-kidney-metabolic conditions.

The primary model is a locked histogram gradient boosting model developed in CHARLS and externally validated in HRS, ELSA, SHARE, and KLoSA. Risk groups use CHARLS-derived fixed thresholds:

- Low risk: predicted risk < 1.61%
- Intermediate risk: 1.61% to < 3.75%
- High risk: predicted risk >= 3.75%

The prediction horizon is the interval from the current assessment to the next study follow-up. It is not a fixed 1-year or 2-year mortality risk.

## Files

- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
- `risk_tools/locked_hgb_death_model.joblib`: locked prediction model.
- `risk_tools/risk_tool_feature_schema.json`: predictor list, coding rules, and risk thresholds.
- `risk_tools/example_risk_inputs.csv`: example input file for batch scoring.

## Local Use

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud Deployment

1. Create a public GitHub repository.
2. Upload the contents of this folder to the repository root.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Select the repository and set the main file path to `app.py`.
5. Deploy the app and use the generated public URL in the manuscript or supplement.

## Hugging Face Spaces Deployment

1. Create a new Space at https://huggingface.co/spaces.
2. Select `Streamlit` as the Space SDK.
3. Upload `app.py`, `requirements.txt`, and the `risk_tools/` folder.
4. The Space will build automatically and provide a public URL.

## Suggested Manuscript Wording

The locked model was implemented as a browser-based risk calculator. The application accepts harmonized clinical and functional predictors, returns an individualized predicted probability of mortality before the next study follow-up, and assigns risk groups using CHARLS-derived thresholds. The calculator and model files are available at [URL to be inserted on deployment].

## Important Limitations

This application is intended for research use. It should not be used as a standalone clinical decision-making tool. The model estimates risk before the next follow-up visit in longitudinal aging cohorts and does not provide fixed-year mortality prediction.
