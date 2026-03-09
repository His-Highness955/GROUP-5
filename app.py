import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Page Config ---
st.set_page_config(page_title="CVD Risk Predictor", layout="wide", page_icon="❤️")

# --- Custom CSS for BOUESTI Branding ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Note: Ensure your 'final_ridge_cvd_model.pkl' was saved as a Pipeline object
    # containing both the preprocessor and the Ridge model.
    try:
        return joblib.load('final_ridge_cvd_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'final_ridge_cvd_model.pkl' is in the same directory.")
        return None

model = load_model()

# --- Header ---
st.title("❤️ Heart Disease & Stroke Risk Predictor")
st.info("Educational Tool: Predicting Cardiovascular Disease (CVD) risks using Ridge Regression.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("👤 Patient Demographics")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 45)
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    
    st.header("🏥 Clinical Data")
    hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 105.0)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 24.5)
    
    st.header("🚬 Lifestyle")
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Feature Engineering Function ---
def engineer_features(age, glucose, bmi_val):
    # Age Group
    if age <= 18: age_grp = 'child'
    elif age <= 40: age_grp = 'young_adult'
    elif age <= 60: age_grp = 'middle_age'
    else: age_grp = 'senior'
    
    # Glucose Group
    if glucose <= 100: glu_grp = 'normal'
    elif glucose <= 126: glu_grp = 'prediabetes'
    else: glu_grp = 'diabetes' # Simplified for the model
    
    # BMI Group
    if bmi_val < 18.5: bmi_grp = 'underweight'
    elif bmi_val < 25: bmi_grp = 'normal'
    elif bmi_val < 30: bmi_grp = 'overweight'
    else: bmi_grp = 'obese'
    
    return age_grp, glu_grp, bmi_grp

# --- Prediction Logic ---
if st.button("Analyze Risk Profile", type="primary"):
    if model:
        age_group, glucose_group, bmi_group = engineer_features(age, avg_glucose_level, bmi)
        
        # Build Input DataFrame (Ensure column names match the training set EXACTLY)
        input_df = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type], # Capital R to match standard datasets
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
            'age_group': [age_group],
            'glucose_group': [glucose_group],
            'bmi_group': [bmi_group]
        })

        # Generate Prediction
        prediction = model.predict(input_df)[0]
        
        # Check if model supports decision_function (Standard for Ridge)
        try:
            score = model.decision_function(input_df)[0]
        except:
            score = 0.0 # Fallback

        # --- Display Results ---
        st.subheader("Results Analysis")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Risk Decision Score", f"{score:.3f}")
            if prediction == 0:
                st.success("✅ **Low Risk Detected**")
            else:
                st.error("⚠️ **Elevated Risk Detected**")

        with res_col2:
            # Interpretation of the Ridge Score
            if prediction == 0:
                st.write("The model suggests the patient is currently below the threshold for clinical intervention.")
            elif score > 2.0:
                st.warning("**High Alert:** High risk for both Heart Disease and Stroke.")
            elif score > 1.0:
                st.warning("**Heart Priority:** Signal is stronger for Heart Disease.")
            else:
                st.warning("**Stroke Priority:** Signal is stronger for Cerebrovascular issues.")

        # Data Preview
        with st.expander("View Processed Feature Data"):
            st.table(input_df)
    else:
        st.warning("Please upload or link your model to proceed.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px 0;'>
        <strong>BOUESTI Engineering Project</strong> • Ikere-Ekiti / Benin City • March 2026<br>
        <small>Ridge Regression Analysis for Cardiovascular Health</small>
    </div>
    """,
    unsafe_allow_html=True
)