# Deployment Checklist

## Option A: Streamlit Community Cloud

1. Create a GitHub repository, for example `ckm-mortality-risk-calculator`.
2. Upload all files in this folder to the repository root.
3. Go to `https://share.streamlit.io/`.
4. Choose the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.
7. Copy the public Streamlit URL into the manuscript, supplement, and code availability statement.

## Option B: Hugging Face Spaces

1. Create a new Space at `https://huggingface.co/spaces`.
2. Select `Streamlit` as the SDK.
3. Upload all files in this folder.
4. Wait for the Space to build.
5. Copy the public Space URL into the manuscript.

## Pre-Submission Checks

- Confirm the public URL opens without login.
- Test the single-person calculator.
- Upload `risk_tools/example_risk_inputs.csv` in the Batch CSV tab and confirm scored output is downloadable.
- Confirm the page states that the tool is for research use and is not a fixed-year mortality risk calculator.
- Archive the exact GitHub commit or release used for submission.

## Manuscript URL Placeholder

Replace this placeholder after deployment:

`https://<your-public-app-url>`
