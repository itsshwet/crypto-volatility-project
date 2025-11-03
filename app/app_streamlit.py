import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load the trained model
# -----------------------------
import os
import joblib

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_vol_model.joblib')
model = joblib.load(model_path)

st.set_page_config(page_title="Crypto Volatility Predictor", layout="wide")
st.title("💹 Crypto Volatility Prediction App")

st.markdown("""
This app predicts the **next 7-day volatility** of a cryptocurrency  
based on recent technical indicators.
""")

# -----------------------------
# 2️⃣ Sidebar input section
# -----------------------------
st.sidebar.header("Input Feature Values")
st.sidebar.markdown("Enter the latest data values below:")

# User inputs
volatility_7d = st.sidebar.number_input("7-Day Volatility", min_value=0.0, max_value=1.0, value=0.3)
volatility_21d = st.sidebar.number_input("21-Day Volatility", min_value=0.0, max_value=1.0, value=0.25)
log_return = st.sidebar.number_input("Log Return", min_value=-0.2, max_value=0.2, value=0.01)
ma_7 = st.sidebar.number_input("7-Day Moving Average", min_value=0.0, value=40000.0)
ma_21 = st.sidebar.number_input("21-Day Moving Average", min_value=0.0, value=38000.0)
vol_ratio = st.sidebar.number_input("Volume Ratio", min_value=0.0, max_value=3.0, value=1.2)

# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    'volatility_7d': [volatility_7d],
    'volatility_21d': [volatility_21d],
    'log_return': [log_return],
    'ma_7': [ma_7],
    'ma_21': [ma_21],
    'vol_ratio': [vol_ratio]
})

st.subheader("🔢 Input Summary")
st.write(input_data)

# -----------------------------
# 3️⃣ Make prediction
# -----------------------------
if st.button("Predict Volatility"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted 7-Day Volatility: **{prediction:.4f}**")

    # Interpretation
    if prediction < 0.2:
        st.info("Market seems **Stable / Low Volatility** 🟢")
    elif prediction < 0.5:
        st.warning("Market showing **Moderate Volatility** 🟠")
    else:
        st.error("Market is **Highly Volatile** 🔴")

    # -----------------------------
    # 4️⃣ Visualization
    # -----------------------------
    st.subheader("📈 Prediction Visualization")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Predicted Volatility"], [prediction], color=['skyblue'])
    ax.set_ylim(0, 1)
    plt.ylabel("Volatility Level")
    plt.title("Predicted 7-Day Volatility")
    st.pyplot(fig)

else:
    st.info("Click **Predict Volatility** after entering values.")

# -----------------------------
# 5️⃣ Footer
# -----------------------------
st.markdown("---")
st.caption("Built by Shwet Anand |  Data Analyst Project | © 2025")
