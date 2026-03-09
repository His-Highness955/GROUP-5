import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Page Config ---
st.set_page_config(page_title="CVD Risk Predictor", layout="wide", page_icon="❤️")

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

# --- Main App Content ---
if not st.session_state.logged_in:
    login_portal()
else:
    # --- Custom CSS ---
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        </style>
        """, unsafe_allow_html=True)

    # --- Model Loading ---
    @st.cache_resource
    def load_model():
        try:
            return joblib.load('final_ridge_cvd_model.pkl')
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'final_ridge_cvd_model.pkl' is in the directory.")
            return None

    model = load_model()

    # --- Header ---
    st.title("🫀 Heart Disease & Stroke Risk Predictor")
    st.markdown("### 🏥 EKITI STATE BOUESTI STUDENT GROUP 5")
    st.info("Educational Tool: Predicting Cardiovascular Disease (CVD) risks using Ridge Regression.")

    # Sidebar Logout Button
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("👤 Patient Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Enter Age", min_value=1, max_value=100, value=45)
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Clinical Data")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=105.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.5)
        
        st.header("🚬 Lifestyle")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # --- Feature Engineering Function ---
    def engineer_features(age, glucose, bmi_val):
        if age <= 18: age_grp = 'child'
        elif age <= 40: age_grp = 'young_adult'
        elif age <= 60: age_grp = 'middle_age'
        else: age_grp = 'senior'
        
        if glucose <= 100: glu_grp = 'normal'
        elif glucose <= 126: glu_grp = 'prediabetes'
        else: glu_grp = 'diabetes'
        
        if bmi_val < 18.5: bmi_grp = 'underweight'
        elif bmi_val < 25: bmi_grp = 'normal'
        elif bmi_val < 30: bmi_grp = 'overweight'
        else: bmi_grp = 'obese'
        
        return age_grp, glu_grp, bmi_grp

    # --- Prediction Logic ---
    if st.button("Analyze Risk Profile", type="primary"):
        if model:
            age_group, glucose_group, bmi_group = engineer_features(age, avg_glucose_level, bmi)
            
            input_df = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status],
                'age_group': [age_group],
                'glucose_group': [glucose_group],
                'bmi_group': [bmi_group]
            })

            # Inference Pipeline
            prediction = model.predict(input_df)[0]
            try:
                score = model.decision_function(input_df)[0]
            except:
                score = 0.0

            st.subheader("Results Analysis")
            # 
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Risk Decision Score", f"{score:.3f}")
                if prediction == 0:
                    st.success("✅ **Low Risk Detected**")
                else:
                    st.error("⚠️ **Elevated Risk Detected**")

            with res_col2:
                if prediction == 0:
                    st.write("The model suggests the patient is currently below the threshold for clinical intervention.")
                elif score > 2.0:
                    st.warning("**High Alert:** High risk for both Heart Disease and Stroke.")
                elif score > 1.0:
                    st.warning("**Heart Priority:** Signal is stronger for Heart Disease.")
                else:
                    st.warning("**Stroke Priority:** Signal is stronger for Cerebrovascular issues.")

            with st.expander("View Processed Feature Data"):
                st.table(input_df)
        else:
            st.warning("Please link your model to proceed.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 20px 0;'>
            <strong>BOUESTI GROUP 5 Project</strong> • EKITI STATE UNIVERSITY TEACHING HOSPITAL<br>
            Ikere-Ekiti / Ikere City • March 2026<br>
            <small>Ridge Regression Analysis for Cardiovascular Health</small>
        </div>
        """,
        unsafe_allow_html=True
    )
