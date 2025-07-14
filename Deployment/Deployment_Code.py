import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Load the trained model ---
try:
    model = joblib.load('diabetes_prediction_pipeline_V2.pkl')
except FileNotFoundError:
    st.error("Error: 'diabetes_prediction_pipeline_V2.pkl' not found. "
             "Please make sure the model file is in the same directory as this script.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("Diabetes Prediction App")
st.markdown("""
    Enter the patient's information below to get a prediction on whether they might have diabetes.
""")

# --- Input Fields ---

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ('Female', 'Male', 'Other'))
    age = st.slider("Age", 0.08, 100.0, 30.0, help="Age of the patient in years")
    hypertension = st.radio("Hypertension", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True,
                            help="0: No hypertension, 1: Has hypertension")

with col2:
    heart_disease = st.radio("Heart Disease", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True,
                             help="0: No heart disease, 1: Has heart disease")
    smoking_history = st.selectbox("Smoking History", ('never', 'No Info', 'current', 'former', 'ever', 'not current'))
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                          help="Body Mass Index (e.g., 25.0 for normal weight)")

st.subheader("Medical Readings")
col3, col4 = st.columns(2)

with col3:
    HbA1c_level = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, step=0.1,
                                  help="Glycated Hemoglobin level (e.g., 5.7 for normal)")
with col4:
    blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=70, max_value=300, value=120, step=1,
                                          help="Fasting Blood Glucose Level (e.g., 120 mg/dL)")


# --- Prediction Button ---
if st.button("Predict Diabetes"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history,
                                  bmi, HbA1c_level, blood_glucose_level]],
                              columns=['gender', 'age', 'hypertension', 'heart_disease',
                                       'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Based on the provided information, the model predicts that the patient **has Diabetes**.")
        st.markdown(f"**Confidence (Diabetes):** {prediction_proba[1]*100:.2f}%")
    else:
        st.success("Based on the provided information, the model predicts that the patient **does NOT have Diabetes**.")
        st.markdown(f"**Confidence (No Diabetes):** {prediction_proba[0]*100:.2f}%")

    st.write("---")
    st.info("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. Always consult with a healthcare professional for diagnosis and treatment.")
