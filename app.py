import streamlit as st
import numpy as np
import joblib

model = joblib.load('ecg_model.pkl')

st.title("ECG Heart Disease Detection")

st.write("Enter ECG Features")

mean = st.number_input("Mean")
std = st.number_input("STD")
maximum = st.number_input("Maximum")
minimum = st.number_input("Minimum")
energy = st.number_input("Energy")

if st.button("Predict"):
    features = np.array([[mean, std, maximum, minimum, energy]])
    prediction = model.predict(features)[0]

    if prediction == 0:
        result = "✅ Normal Heart (No Disease)"
    else:
        result = "⚠️ Heart Disease Detected"

    st.success(result)
