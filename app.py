import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# Header
st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts if a patient has heart disease based on clinical parameters.
Data based on: Cleveland, Hungary, and Statlog datasets.
""")

# Sidebar for inputs
st.sidebar.header("Patient Data Input")

def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    
    cp_options = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'}
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
    
    resting_blood_pressure = st.sidebar.number_input("Resting Blood Pressure (mm/Hg)", 50, 250, 120)
    cholesterol = st.sidebar.number_input("Serum Cholestrol (mg/dl)", 100, 600, 200)
    fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x==1 else "False")
    
    ecg_options = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
    rest_ecg = st.sidebar.selectbox("Resting ECG Results", options=list(ecg_options.keys()), format_func=lambda x: ecg_options[x])
    
    max_heart_rate_achieved = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exercise_induced_angina = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    st_depression = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
    
    slope_options = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
    st_slope = st.sidebar.selectbox("ST Slope", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'rest_ecg': rest_ecg,
        'max_heart_rate_achieved': max_heart_rate_achieved,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'st_slope': st_slope
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Area
st.subheader("Patient Parameters")
st.write(input_df)

# Prediction
if st.button("Predict Result"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 Higher Risk of Heart Disease")
    else:
        st.success("✅ Normal / Low Risk")

    st.subheader("Confidence Level")
    st.write(f"Probability of Heart Disease: **{prediction_proba[0][1]:.2%}**")

# Footer/About
st.info("Note: This is for educational purposes only. Consult a doctor for medical advice.")