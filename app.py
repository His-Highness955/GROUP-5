import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="CVD Risk Predictor", layout="wide", page_icon="🫀")

# --- Session State for Login ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Portal Logic ---
def login_portal():
    st.title("🔐 Secure Access")
    st.info("Please log in to access the CVD Risk Predictor.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "team 5" and password == "bouesti2026":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Username or Password")

# --- Function to Save Data ---
def save_patient_data(patient_name, input_df, prediction, score):
    file_path = 'patient_records.csv'
    data_to_save = input_df.copy()
    data_to_save['patient_name'] = patient_name
    data_to_save['prediction'] = prediction
    data_to_save['score'] = score
    data_to_save['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(file_path):
        data_to_save.to_csv(file_path, index=False)
    else:
        data_to_save.to_csv(file_path, mode='a', header=False, index=False)

# --- Feature Engineering ---
def engineer_features(age, glucose, bmi_val):
    age_grp = 'child' if age <= 18 else 'young_adult' if age <= 40 else 'middle_age' if age <= 60 else 'senior'
    glu_grp = 'normal' if glucose <= 100 else 'prediabetes' if glucose <= 126 else 'diabetes'
    bmi_grp = 'underweight' if bmi_val < 18.5 else 'normal' if bmi_val < 25 else 'overweight' if bmi_val < 30 else 'obese'
    return age_grp, glu_grp, bmi_grp

# --- Main App Content ---
if not st.session_state.logged_in:
    login_portal()
else:
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        </style>
        """, unsafe_allow_html=True)

    model = joblib.load('final_ridge_cvd_model.pkl') if os.path.exists('final_ridge_cvd_model.pkl') else None

    st.title("🫀 Heart Disease & Stroke Risk Predictor")
    st.markdown("### 🏥 EKITI STATE BOUESTI STUDENT GROUP 5 CLINIC")

    with st.sidebar:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.header("👤 Patient Info")
        patient_name = st.text_input("Patient Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Enter Age", min_value=1, max_value=100, value=45)
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Clinical Data")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 105.0)
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 24.5)
        
        st.header("🚬 Lifestyle")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked", "Student"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Prediction Logic
    if st.button("Analyze & Save Profile", type="primary"):
        if model and patient_name:
            age_grp, glu_grp, bmi_grp = engineer_features(age, avg_glucose_level, bmi)
            input_df = pd.DataFrame({
                'gender': [gender], 'age': [age], 'hypertension': [hypertension], 
                'ever_married': [ever_married], 'work_type': [work_type], 
                'Residence_type': [residence_type], 'avg_glucose_level': [avg_glucose_level], 
                'bmi': [bmi], 'smoking_status': [smoking_status], 
                'age_group': [age_grp], 'glucose_group': [glu_grp], 'bmi_group': [bmi_grp]
            })
            
            prediction = model.predict(input_df)[0]
            score = model.decision_function(input_df)[0] if hasattr(model, 'decision_function') else 0.0

            save_patient_data(patient_name, input_df, prediction, score)
            
            st.success(f"Risk analysis complete for {patient_name}!")
            
            st.metric("Risk Decision Score", f"{score:.3f}")
            
            if score > 2.0: st.error("⚠️ **High Alert:** High risk for both Heart Disease and Stroke.")
            elif score > 1.0: st.warning("⚠️ **Priority:** Elevated risk detected.")
            else: st.success("✅ **Low Risk Detected.**")
        else:
            st.warning("Please check model connection and enter patient name.")

    with st.expander("View Saved Patient Records 📝 (Admin)"):
        if os.path.exists('patient_records.csv'):
            st.dataframe(pd.read_csv('patient_records.csv'))
            st.download_button("Download Records", data=open('patient_records.csv', 'rb'), file_name='patient_records.csv')
        else:
            st.write("No records found.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>BOUESTI GROUP 5 Project • March 2026</div>", unsafe_allow_html=True)
