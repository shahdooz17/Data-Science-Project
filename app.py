import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "best_final_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

THRESHOLD = 0.3

# -----------------------------
# Prediction Function
# -----------------------------
def make_prediction(df):
    probs = model.predict(df)
    preds = (probs > THRESHOLD).astype(int)
    return preds, probs


# ============================================================
# Streamlit UI
# ============================================================

st.title("💳 Fraud Detection System")
st.subheader("Upload a processed CSV file and get predictions instantly.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("### 🔍 Data Preview (First 100 rows only)")
    st.dataframe(data.head(100))

    if st.button("🔮 Predict Fraud"):
        preds, probs = make_prediction(data)

        data["Fraud_Probability"] = probs
        data["Prediction"] = preds
        data["Label"] = data["Prediction"].map({0: "Not Fraud", 1: "Fraud"})

        st.success("Prediction Completed!")

        # Show mini preview
        st.write("### 🔎 Sample Results")
        st.dataframe(data.head(50))

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Predictions CSV",
            data=csv,
            file_name="Fraud_Predictions.csv",
            mime="text/csv"
        )
