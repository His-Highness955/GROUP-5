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

# --- Robust Data Persistence ---
def save_patient_data(patient_name, input_df, pred_type, score, risk_lvl):
    file_path = 'patient_records.csv'
    new_record = input_df.copy()
    new_record['patient_name'] = patient_name
    new_record['prediction_type'] = pred_type
    new_record['score'] = score
    new_record['risk_level'] = risk_lvl
    new_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cols = ['gender', 'age', 'hypertension', 'ever_married', 'work_type', 
            'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 
            'age_group', 'glucose_group', 'bmi_group', 'patient_name', 
            'prediction_type', 'score', 'risk_level', 'timestamp']
    
    new_record = new_record[cols]
    file_exists = os.path.exists(file_path)
    new_record.to_csv(file_path, mode='a', header=not file_exists, index=False)

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
    model = joblib.load('final_ridge_cvd_model.pkl') if os.path.exists('final_ridge_cvd_model.pkl') else None

    st.title("🫀 GROUP E Risk prediction Portal")
    st.markdown("### 🏥 EKITI STATE BOUESTI STUDENT GROUP 5 CLINIC")

    with st.sidebar:
        if st.button("Logout"): st.session_state.logged_in = False; st.rerun()
        st.header("👤 Patient Info")
        patient_name = st.text_input("Patient Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1, 100, 45)
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        st.header("🏥 Clinical Data")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 105.0)
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 24.5)
        st.header("🚬 Lifestyle")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked", "Student", "jobless"])
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
            age_grp, glu_grp, bmi_grp = engineer_features(age, avg_glucose_level, bmi)
            input_df = pd.DataFrame({
                'gender': [gender], 'age': [age], 'hypertension': [hypertension], 
                'ever_married': [ever_married], 'work_type': [work_type], 
                'Residence_type': [residence_type], 'avg_glucose_level': [avg_glucose_level], 
                'bmi': [bmi], 'smoking_status': [smoking_status], 
                'age_group': [age_grp], 'glucose_group': [glu_grp], 'bmi_group': [bmi_grp]
            })
            
            raw_score = model.decision_function(input_df)[0]
            multipliers = {"Heart": 0.9, "Stroke": 1.1, "Combined": 1.4}
            adj_score = raw_score * multipliers.get(pred_type, 1.0)
            risk_lvl = "CRITICAL" if adj_score > 2.0 else "ELEVATED" if adj_score > 1.0 else "STABLE"
            save_patient_data(patient_name, input_df, pred_type, adj_score, risk_lvl)
            
            st.divider()
            st.metric(f"{pred_type} Risk Score", f"{adj_score:.3f}")
            if risk_lvl == "CRITICAL": st.error(f"Status: {risk_lvl}")
            elif risk_lvl == "ELEVATED": st.warning(f"Status: {risk_lvl}")
            else: st.success(f"Status: {risk_lvl}")

            # --- Personalized Patient Risk Contribution ---
            st.subheader("📊 Personalized Risk Drivers")
            try:
                # Automagically find the model step and preprocessor
                model_step = None
                preprocessor = None
                
                if hasattr(model, 'named_steps'):
                    steps = list(model.named_steps.values())
                    model_step = steps[-1]
                    if len(steps) > 1:
                        preprocessor = steps[0]
                elif hasattr(model, 'coef_'):
                    model_step = model

                if model_step and hasattr(model_step, 'coef_'):
                    # Transform input to match model space
                    if preprocessor and hasattr(preprocessor, 'transform'):
                        transformed_data = preprocessor.transform(input_df)
                    else:
                        transformed_data = input_df.select_dtypes(include=[np.number]).values

                    coefs = model_step.coef_.flatten()
                    # Mapping numerical inputs
                    features = ['Age', 'Hypertension', 'Glucose', 'BMI']
                    n_feats = min(len(features), transformed_data.shape[1])
                    
                    impacts = [transformed_data[0, i] * coefs[i] for i in range(n_feats)]
                    weights = pd.DataFrame({'Feature': features[:n_feats], 'Contribution': impacts})
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    sns.barplot(x='Contribution', y='Feature', data=weights, palette='coolwarm')
                    st.pyplot(fig2)
                    st.caption("This shows the scaled impact of your clinical metrics on the risk score.")
                else:
                    st.error("Could not process model coefficients.")
            except Exception as e:
                st.error(f"Debug: Calculation error: {e}")

    with st.expander("View Saved Patient Records 📝 (Admin)"):
        if os.path.exists('patient_records.csv'):
            try:
                st.dataframe(pd.read_csv('patient_records.csv', engine='python'))
            except Exception: st.error("File is corrupted.")
        if st.button("⚠️ Reset Records File"):
            if os.path.exists('patient_records.csv'):
                os.remove('patient_records.csv')
                st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>BOUESTI GROUP 5 Project • March 2026 • ikere ekiti</div>", unsafe_allow_html=True)


