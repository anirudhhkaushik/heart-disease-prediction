import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load model
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

# Sidebar
st.sidebar.header("Patient Data Input")

def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")

    cp_options = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'}
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", list(cp_options.keys()), format_func=lambda x: cp_options[x])

    resting_blood_pressure = st.sidebar.number_input("Resting Blood Pressure", 50, 250, 120)
    cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])

    ecg_options = {0: 'Normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'}
    rest_ecg = st.sidebar.selectbox("Rest ECG", list(ecg_options.keys()), format_func=lambda x: ecg_options[x])

    max_heart_rate_achieved = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exercise_induced_angina = st.sidebar.selectbox("Exercise Angina", [0, 1])
    st_depression = st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0)

    slope_options = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
    st_slope = st.sidebar.selectbox("ST Slope", list(slope_options.keys()), format_func=lambda x: slope_options[x])

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

    return pd.DataFrame([data])

input_df = user_input_features()

# Show input
st.subheader("Patient Parameters")
st.write(input_df)

# Prediction button
if st.button("Predict Result"):

   
    input_df = pd.get_dummies(input_df)

    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # ✅ Prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Output
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 Higher Risk of Heart Disease")
    else:
        st.success("✅ Normal / Low Risk")

    # Confidence
    st.subheader("Confidence Level")
    st.write(f"Probability of Heart Disease: **{prediction_proba[0][1]:.2%}**")

# Footer
st.info("Note: This is for educational purposes only.")