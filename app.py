import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Config & Session State ---
st.set_page_config(page_title="CVD Clinical Portal", layout="wide", page_icon="🫀")
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

# --- Login & Data Persistence ---
def login_portal():
    st.title("🔐 Secure Clinical Access")
    if st.text_input("Username") == "team 5" and st.text_input("Password", type="password") == "bouesti2026":
        if st.button("Login"): st.session_state.logged_in = True; st.rerun()

def save_patient_data(patient_name, input_df, pred_type, score, risk_level):
    file_path = 'patient_records.csv'
    data = input_df.copy()
    data.update({'patient_name': patient_name, 'prediction_type': pred_type, 
                 'score': score, 'risk_level': risk_level, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    
    if not os.path.exists(file_path): data.to_csv(file_path, index=False)
    else: data.to_csv(file_path, mode='a', header=False, index=False)

def engineer_features(age, glucose, bmi_val):
    age_grp = 'child' if age <= 18 else 'young_adult' if age <= 40 else 'middle_age' if age <= 60 else 'senior'
    glu_grp = 'normal' if glucose <= 100 else 'prediabetes' if glucose <= 126 else 'diabetes'
    bmi_grp = 'underweight' if bmi_val < 18.5 else 'normal' if bmi_val < 25 else 'overweight' if bmi_val < 30 else 'obese'
    return age_grp, glu_grp, bmi_grp

# --- Main App ---
if not st.session_state.logged_in:
    login_portal()
else:
    model = joblib.load('final_ridge_cvd_model.pkl') if os.path.exists('final_ridge_cvd_model.pkl') else None
    
    st.title("🫀 Heart Disease & Stroke Risk Predictor")
    
    with st.sidebar:
        patient_name = st.text_input("Patient Full Name")
        age = st.number_input("Age", 1, 100, 45)
        glucose = st.number_input("Glucose", 50.0, 300.0, 105.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
        hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Student", "Other"])

    # --- Clinical Assessment Engine ---
    st.subheader("Select Assessment Type")
    col1, col2, col3 = st.columns(3)

    def run_prediction(pred_type):
        if not model or not patient_name: return st.warning("Ensure model is loaded and name entered.")
        
        # Prepare Data
        age_grp, glu_grp, bmi_grp = engineer_features(age, glucose, bmi)
        input_df = pd.DataFrame({'age': [age], 'hypertension': [hypertension], 'avg_glucose_level': [glucose], 
                                 'bmi': [bmi], 'work_type': [work_type], 'age_group': [age_grp]})
        
        # Hospital Standard: Apply multipliers for specific conditions
        raw_score = model.decision_function(input_df)[0]
        if pred_type == "Heart": adj_score = raw_score * 0.9
        elif pred_type == "Stroke": adj_score = raw_score * 1.1
        else: adj_score = raw_score * 1.4 # Combined
        
        risk_level = "CRITICAL" if adj_score > 2.0 else "ELEVATED" if adj_score > 1.0 else "STABLE"
        save_patient_data(patient_name, input_df, pred_type, adj_score, risk_level)
        
        # UI Results
        st.metric(f"{pred_type} Risk Score", f"{adj_score:.3f}")
        st.info(f"Clinical Status: **{risk_level}**")
        
        # Probability Breakdown
        st.table(pd.DataFrame({"Metric": ["Condition", "Risk Index", "Triage"], 
                               "Value": [pred_type, f"{max(0, adj_score*15):.1f}%", risk_level]}))

    if col1.button("Predict Heart Risk"): run_prediction("Heart")
    if col2.button("Predict Stroke Risk"): run_prediction("Stroke")
    if col3.button("Predict Both"): run_prediction("Combined")

    if os.path.exists('patient_records.csv'):
        with st.expander("View Clinical Logs"):
            st.dataframe(pd.read_csv('patient_records.csv'))
