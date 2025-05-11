import streamlit as st
import joblib
import numpy as np
import os

# Load the scaler
scaler_path = "model/scaler.pkl"
if not os.path.exists(scaler_path):
    st.error("Scaler file not found at 'model/scaler.pkl'.")
else:
    scaler = joblib.load(scaler_path)
    st.success("Scaler loaded successfully.")

    # Example input features (replace with your actual features)
    st.header("Heart Prediction Example")
    age = st.number_input("Age", 18, 100, 50)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    thalach = st.number_input("Max Heart Rate", 60, 200, 150)

    # Button to apply scaling
    if st.button("Scale Input"):
        input_data = np.array([[age, chol, thalach]])
        scaled_data = scaler.transform(input_data)
        st.write("Scaled Data:", scaled_data)
