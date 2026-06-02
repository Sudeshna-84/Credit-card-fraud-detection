import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")
st.title("🚨 Credit Card Fraud Detection — Demo App")

MODEL_DIR = "models"
pipeline_path = os.path.join(MODEL_DIR, "final_pipeline.joblib")

# ------------------------------
# Load the trained pipeline
# ------------------------------
if not os.path.exists(pipeline_path):
    st.error("❌ final_pipeline.joblib not found in models/. Please run the Jupyter Notebook to train and save the model.")
    st.stop()
else:
    pipe = joblib.load(pipeline_path)
    st.success("✅ Model loaded successfully!")

# ----------------------------------
# Get expected feature names
# ----------------------------------
expected_features = None
if hasattr(pipe, "named_steps"):
    model_step = pipe.named_steps.get("model", None)
    scaler_step = pipe.named_steps.get("scaler", None)

# Extract feature names from scaler (most reliable)
if hasattr(scaler_step, "feature_names_in_"):
    expected_features = list(scaler_step.feature_names_in_)
else:
    st.warning("⚠ Could not detect expected feature names. Make sure your CSV has same columns as training data.")

# -----------------------------------------------------
# 1. BULK CSV PREDICTION
# -----------------------------------------------------
st.header("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV with transaction data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Uploaded Data Preview")
    st.write(df.head())

    # ------------------------------
    # Remove Class column if present
    # ------------------------------
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
        st.info("ℹ️ 'Class' column detected and removed for prediction.")

    # ------------------------------
    # Align CSV features to expected model features
    # ------------------------------
    if expected_features is not None:
        missing = set(expected_features) - set(df.columns)
        extra = set(df.columns) - set(expected_features)

        if missing:
            st.error(f"❌ Missing columns required for prediction: {missing}")
            st.stop()

        if extra:
            st.warning(f"⚠ Extra columns found and ignored: {extra}")
            df = df[expected_features]  # Keep only expected columns

    # ------------------------------
    # Run prediction
    # ------------------------------
    if st.button("🔍 Run Prediction"):
        X = df.copy()
        preds = pipe.predict(X)
        probs = None

        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X)[:, 1]

        # Build result output
        output = df.copy()
        output["predicted_label"] = preds

        if probs is not None:
            output["fraud_probability"] = probs

        st.subheader("📊 Prediction Output")
        st.dataframe(output.head(20))

        # Download button
        csv_out = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions",
            data=csv_out,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

# -----------------------------------------------------
# 2. SINGLE TRANSACTION PREDICTION (Optional)
# -----------------------------------------------------
st.header("🧪 Single Transaction Prediction")

st.write("Enter values manually (Only works if feature names are known).")

if expected_features:
    single_input = {}
    for feature in expected_features:
        single_input[feature] = st.number_input(feature, value=0.0)

    if st.button("🔮 Predict Single Transaction"):
        X_single = pd.DataFrame([single_input])
        pred = pipe.predict(X_single)[0]
        prob = pipe.predict_proba(X_single)[0][1] if hasattr(pipe, "predict_proba") else None

        st.success(f"Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
        if prob is not None:
            st.info(f"Fraud Probability: {prob:.4f}")
else:
    st.warning("⚠ Single transaction input disabled because expected feature list could not be detected.")
