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
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Portal Logic ---
def login_portal():
    # CSS for watermark
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                        url("ccoeikere.png");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=150)
    
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🫀 CVD Risk Prediction Portal</h1>
            <h3>BOUESTI CIS STUDENT GROUP 5 CLINIC</h3>
            <p>Authorized personnel only. Please enter your credentials to proceed.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Log in with your Team 5 credentials.")
        username = st.text_input("Username", value="team 5")
        password = st.text_input("Password", type="password")
        
        if st.button("Login 🔐"):
            if username == "team 5" and password == "bouesti2026":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password")
                
    st.markdown("---")
    st.caption("Developed for the CIS Department Coursework under the supervision of Mrs. T.O. Adefehinti.")

# --- Data Persistence ---
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

# --- Feature Engineering ---
def engineer_features(age, glucose, bmi_val, hypertension, diabetes):
    if age <= 35: age_grp = 'young'
    elif age <= 55: age_grp = 'middle'
    else: age_grp = 'senior'
    
    if bmi_val < 18.5: bmi_grp = 'underweight'
    elif bmi_val < 25: bmi_grp = 'normal'
    elif bmi_val < 30: bmi_grp = 'overweight'
    else: bmi_grp = 'obese'
    
    if glucose < 100: glu_grp = 'normal'
    elif glucose < 126: glu_grp = 'prediabetes'
    else: glu_grp = 'diabetes'
    
    return age_grp, glu_grp, bmi_grp

# --- Main App ---
if not st.session_state.logged_in:
    login_portal()
else:
    model = joblib.load('final_ridge_cvd_model.pkl') if os.path.exists('final_ridge_cvd_model.pkl') else None

    st.title("🫀 GROUP E HEART OR STROKE Risk prediction Portal")
    st.markdown("### 🏥 EKITI STATE BOUESTI CIS STUDENT GROUP 5 CLINIC")

    with st.sidebar:
        if st.button("Logout"): 
            st.session_state.logged_in = False
            st.rerun()
        st.header("👤 Patient Info")
        patient_name = st.text_input("Patient Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 0, 120, 45)
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Primary Clinical Data")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 250, 120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        
        diabetes = st.radio("Diabetes / Hyperglycemia?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        dyslipidemia = st.radio("Dyslipidemia (High Cholesterol)?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", 0.0, 120.0, 90.0)
        bmi_input = st.text_input("Body Mass Index (BMI) - Leave blank for Median (28.1)", value="24.5")
        
        st.header("⚠️ Secondary Contributors")
        ckd = st.checkbox("Chronic Kidney Disease (CKD)")
        stress = st.checkbox("High Psychosocial Stress")
        sedentary = st.checkbox("Sedentary Lifestyle (Lack of Exercise)")
        infection = st.checkbox("History of Infections (e.g. Rheumatic Fever)")
        
        st.header("🚬 Lifestyle")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Student", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.subheader("Select what you want to predict")
    col1, col2, col3 = st.columns(3)
    pred_type = None
    if col1.button("Predict Heart Risk"): pred_type = "Heart"
    if col2.button("Predict Stroke Risk"): pred_type = "Stroke"
    if col3.button("Predict Both"): pred_type = "Combined"

    if pred_type:
        if not model or not patient_name:
            st.warning("Please ensure model is loaded and patient name is provided.")
        else:
            bmi = float(bmi_input) if bmi_input.strip() != "" else 28.1
            age_grp, glu_grp, bmi_grp = engineer_features(age, avg_glucose_level, bmi, hypertension, diabetes)
            
            input_df = pd.DataFrame({
                'gender': [gender], 'age': [age], 'hypertension': [hypertension], 
                'ever_married': ["No"], 'work_type': [work_type], 'Residence_type': [residence_type], 
                'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi], 'smoking_status': [smoking_status],
                'age_group': [age_grp], 'glucose_group': [glu_grp], 'bmi_group': [bmi_grp]
            })
            
            bp_boost = 0.0
            if systolic_bp >= 140 or diastolic_bp >= 90: bp_boost = 0.30
            elif systolic_bp >= 130 or diastolic_bp >= 85: bp_boost = 0.15
            
            raw_score = model.decision_function(input_df)[0]
            clinical_boost = 1.0 + (diabetes * 0.25) + (dyslipidemia * 0.20) + (ckd * 0.15) + \
                             (stress * 0.10) + (sedentary * 0.10) + (infection * 0.15) + bp_boost
            
            multipliers = {"Heart": 0.9, "Stroke": 1.1, "Combined": 1.4}
            adj_score = raw_score * multipliers.get(pred_type, 1.0) * clinical_boost
            
            probability = 1 / (1 + np.exp(-adj_score))
            risk_pct = probability * 100
            
            risk_lvl = "CRITICAL" if risk_pct > 75 else "ELEVATED" if risk_pct > 50 else "STABLE"
            save_patient_data(patient_name, input_df, pred_type, adj_score, risk_lvl)
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{adj_score:.2f}")
            c2.metric("Probability", f"{risk_pct:.1f}%")
            c3.metric("Status", risk_lvl)
            
            st.subheader("📊 Clinical Impact Analysis")
            drivers = {
                'Age/Bio': age * 0.01,
                'Primary (HTN/DB/BP)': (hypertension + diabetes + (bp_boost * 2)) * 0.6,
                'Lipids/Dyslipidemia': dyslipidemia * 0.4,
                'Lifestyle/Stress': (stress + sedentary + (1 if smoking_status=="smokes" else 0)) * 0.3,
                'Organ/Infection': (ckd + infection) * 0.4
            }
            driver_df = pd.DataFrame(list(drivers.items()), columns=['Factor', 'Impact'])
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Impact', y='Factor', data=driver_df, palette='OrRd_r')
            st.pyplot(fig)
            
            st.subheader("🩸 Blood Pressure Assessment")
            if systolic_bp >= 180 or diastolic_bp >= 120:
                st.error("🚨 HYPERTENSIVE CRISIS: Immediate medical attention required.")
            elif systolic_bp >= 140 or diastolic_bp >= 90:
                st.warning("⚠️ Stage 2 Hypertension: Consult a physician immediately.")
            elif systolic_bp >= 130 or diastolic_bp >= 80:
                st.info("ℹ️ Stage 1 Hypertension: Lifestyle changes and monitoring advised.")
            else:
                st.success("✅ Blood pressure is within a normal range.")

    with st.expander("Admin 🗃️: Patient Database"):
        if os.path.exists('patient_records.csv'):
            st.dataframe(pd.read_csv('patient_records.csv'))
            if st.button("Clear All Records"):
                os.remove('patient_records.csv')
                st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>• BOUESTI CIS student GROUP 5 Project • </br> An assignment given by MRS T.O. ADEFEHINTI • March 2026 • Ikere-Ekiti</div>", unsafe_allow_html=True)
