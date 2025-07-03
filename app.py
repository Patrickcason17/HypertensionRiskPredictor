import streamlit as st
import joblib
import numpy as np
model = joblib.load("bp_predictor_model.pkl")
st.title("🩺 Hypertension (BP) Risk Predictor")
st.markdown("Enter the details below to check your 10-year risk of hypertension.")
male = st.radio("Gender", ["Male", "Female"])
male = 1 if male == "Male" else 0
age = st.slider("Age", 20, 80)
cigsPerDay = st.slider("Cigarettes per Day", 0, 50)
sysBP = st.slider("Systolic BP", 80, 200)
diaBP = st.slider("Diastolic BP", 50, 130)
bmi = st.slider("Body Mass Index (BMI)", 10.0, 45.0)
glucose = st.slider("Glucose Level", 50, 300)
heartRate = st.slider("Heart Rate", 40, 150)
education = 3
currentSmoker = 0
BPMeds = 0
prevalentStroke = 0
prevalentHyp = 0
diabetes = 0
totChol = 200
input_data = np.array([[male, age, education, currentSmoker, cigsPerDay,
                        BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol,
                        sysBP, diaBP, bmi, heartRate, glucose]])
if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Hypertension – Please consult your doctor.")
    else:
        st.success("✅ Low Risk of Hypertension – Keep up the healthy lifestyle!")
