import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Group 5 Clinical Portal", layout="wide", page_icon="🫀")

# --- Session State ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'pred_type' not in st.session_state: st.session_state.pred_type = None

# --- Helper Functions ---
def save_patient_data(patient_name, input_df, pred_type, score, risk_lvl):
    file_path = 'patient_records.csv'
    new_record = input_df.copy()
    new_record['patient_name'] = patient_name
    new_record['prediction_type'] = pred_type
    new_record['score'] = score
    new_record['risk_level'] = risk_lvl
    new_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(file_path)
    new_record.to_csv(file_path, mode='a', header=not file_exists, index=False)

def engineer_features(age, glucose, bmi_val, hypertension, diabetes):
    age_grp = 'young' if age <= 35 else 'middle' if age <= 55 else 'senior'
    bmi_grp = 'underweight' if bmi_val < 18.5 else 'normal' if bmi_val < 25 else 'overweight' if bmi_val < 30 else 'obese'
    glu_grp = 'normal' if glucose < 100 else 'prediabetes' if glucose < 126 else 'diabetes'
    return age_grp, glu_grp, bmi_grp

# --- UI Components ---
def login_portal():
    st.markdown("""<style>.stApp {background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), url("/app/static/ccoeikere.png"); background-size: contain; background-position: center; background-repeat: no-repeat;}</style>""", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><h1>🫀 CVD Risk Prediction Portal</h1><h3>BOUESTI CIS STUDENT GROUP 5 CLINIC</h3></div>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login 🔐"):
            if username == "team 5" and password == "bouesti2026":
                st.session_state.logged_in = True
                st.rerun()
            else: st.error("Invalid Username or Password")
    st.markdown("<div style='text-align: center; color: #0000ff;'>Developed by CIS student GROUP 5 as model for CSC 309 research given by Mrs. T.O. Adefehinti.</div>", unsafe_allow_html=True)

# --- Main App Execution ---
if not st.session_state.logged_in:
    login_portal()
else:
    model = joblib.load('final_ridge_cvd_model.pkl') if os.path.exists('final_ridge_cvd_model.pkl') else None
    
    st.title("🫀 GROUP E Heart/Stroke Risk Portal")
    
    with st.sidebar:
        if st.button("Logout"): st.session_state.logged_in = False; st.rerun()
        st.header("👤 Patient Info")
        patient_name = st.text_input("Patient Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 0, 120, 45)
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Primary Clinical Data")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 250, 120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        diabetes = st.radio("Diabetes?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", 0.0, 120.0, 90.0)
        bmi_input = st.text_input("BMI", value="24.5")
        
        st.header("🚬 Lifestyle")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Student", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.subheader("Select Prediction Type")
    c1, c2, c3 = st.columns(3)
    if c1.button("Predict Heart Risk"): st.session_state.pred_type = "Heart"
    if c2.button("Predict Stroke Risk"): st.session_state.pred_type = "Stroke"
    if c3.button("Predict Both"): st.session_state.pred_type = "Combined"

    if st.session_state.pred_type:
        st.info(f"Current Prediction Mode: **{st.session_state.pred_type}**")
        
        if not model or not patient_name:
            st.warning("Please ensure model is loaded and patient name is provided.")
        else:
            bmi = float(bmi_input) if bmi_input.strip() != "" else 28.1
            age_grp, glu_grp, bmi_grp = engineer_features(age, avg_glucose_level, bmi, hypertension, diabetes)
            input_df = pd.DataFrame({'gender': [gender], 'age': [age], 'hypertension': [hypertension], 'ever_married': ["No"], 'work_type': [work_type], 'Residence_type': [residence_type], 'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi], 'smoking_status': [smoking_status], 'age_group': [age_grp], 'glucose_group': [glu_grp], 'bmi_group': [bmi_grp]})
            
            bp_boost = 0.30 if (systolic_bp >= 140 or diastolic_bp >= 90) else 0.0
            raw_score = model.decision_function(input_df)[0]
            adj_score = raw_score * {"Heart": 0.9, "Stroke": 1.1, "Combined": 1.4}.get(st.session_state.pred_type, 1.0) * (1 + bp_boost)
            
            risk_pct = (1 / (1 + np.exp(-adj_score))) * 100
            risk_lvl = "CRITICAL" if risk_pct > 75 else "ELEVATED" if risk_pct > 50 else "STABLE"
            
            save_patient_data(patient_name, input_df, st.session_state.pred_type, adj_score, risk_lvl)
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{adj_score:.2f}")
            col2.metric("Probability", f"{risk_pct:.1f}%")
            col3.metric("Status", risk_lvl)

            # 

[Image of heart anatomy and cardiovascular system]

            fig, ax = plt.subplots(figsize=(6, 2))
            sns.barplot(x=['Age', 'BP', 'BMI'], y=[age*0.01, bp_boost, 0.2], palette='OrRd_r')
            st.pyplot(fig)

    with st.expander("Admin 🗃️: Patient Database"):
        if os.path.exists('patient_records.csv'):
            df = pd.read_csv('patient_records.csv')
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data as CSV", csv, "patient_records.csv", "text/csv")
            if st.button("Clear Records"):
                os.remove('patient_records.csv')
                st.rerun()

    st.markdown("<div style='text-align: center; color: #888;'>• Group 5 Project - March 2026 •</div>", unsafe_allow_html=True)
