import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import gdown

# Download model & features if not already present

model_url = "https://drive.google.com/file/d/1bWuKTXBZFuhBNDJYCrYEXJWGb5F49kW2/view?usp=sharing"
features_url= "https://drive.google.com/file/d/1X4LCnWVPqbQ5va-LIk8lZIkn-poLdgXY/view?usp=sharing"

if not os.path.exists("random_forest_model.pkl"):
    gdown.download(model_url, "random_forest_model.pkl", quiet=False)

if not os.path.exists("model_features.pkl"):
    gdown.download(features_url, "model_features.pkl", quiet=False)

# Load model and feature list
model = joblib.load("random_forest_model.pkl")
features = joblib.load("model_features.pkl")

# Set up the Streamlit UI with hospital theme
st.set_page_config(
    page_title="MedPredict - Cancer Survival Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for hospital theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
        padding: 0;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .header-subtitle {
        color: #e2e8f0;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Card styling */
    .info-card {
        background: white;
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #3182ce;
    }

    .input-card {
        background: white;
        color: #2d3748;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }

    .result-card {
        background: white;
        color: #2d3748;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-top: 2rem;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        color: #2d3748;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2f855a 0%, #38a169 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 161, 105, 0.4);
    }

    /* Input styling */
    .stSelectbox > div > div > div {
        background-color: #f7fafc;
        color: #2d3748;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }

    .stNumberInput > div > div > input {
        background-color: #f7fafc;
        color: #2d3748;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }

    .stDateInput > div > div > input {
        background-color: #f7fafc;
        color: #2d3748;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }

    /* Make sure all text in containers is visible */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #2d3748;
    }

    /* Result styling */
    .success-result {
        background: linear-gradient(135deg, #48bb78 0%, #68d391 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }

    .warning-result {
        background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }

    /* Icon styling */
    .icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fef5e7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f6ad55;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #744210;
    }

    /* Medical info cards */
    .medical-info {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #38b2ac;
        color: #234e52;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏥 MedPredict Analytics</h1>
        <p class="header-subtitle">Advanced Cancer Survival Prediction System</p>
    </div>
""", unsafe_allow_html=True)

# Information card
st.markdown("""
    <div class="info-card">
        <div class="medical-info">
            <strong>📊 Clinical Decision Support Tool</strong><br>
            This AI-powered system analyzes multiple clinical parameters to provide survival likelihood estimates. 
            Our model considers treatment timelines, patient demographics, and medical history to generate predictions.
        </div>
    </div>
""", unsafe_allow_html=True)

# Create main input section
st.markdown('<div class="section-header" style="color: white;">📋 Patient Information</div>', unsafe_allow_html=True)

# Patient Demographics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<span style="color: white; font-weight: bold;">👤 Demographics</span>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, key="age")

with col2:
    st.markdown('<span style="color: white; font-weight: bold;">📊 Health Metrics</span>', unsafe_allow_html=True)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5, key="bmi")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=300.0, value=180.0,
                                  key="cholesterol")

with col3:
    st.markdown('<span style="color: white; font-weight: bold;">🚬 Lifestyle Factors</span>', unsafe_allow_html=True)
    smoking_status = st.selectbox("Smoking Status", [
        "Never Smoked", "Passive Smoker", "Former Smoker", "Active Smoker"], key="smoking")
    asthma = st.selectbox("Asthma History", ["No", "Yes"], key="asthma")

st.markdown("---")

# Medical Information
st.markdown('<div class="section-header" style="color: white;">🏥 Medical History & Treatment </div> ', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<span style="color: white; font-weight: bold;">📅 Timeline</span>', unsafe_allow_html=True)
    diagnosis_date = st.date_input("Diagnosis Date", value=datetime(2023, 1, 1), key="diagnosis_date")
    treatment_date = st.date_input("Treatment End Date", value=datetime(2023, 6, 1), key="treatment_date")

with col2:
    st.markdown('<span style="color: white; font-weight: bold;">🔬 Clinical Details</span>', unsafe_allow_html=True)
    cancer_stage = st.selectbox("Cancer Stage", [
        "Stage I", "Stage II", "Stage III", "Stage IV"], key="cancer_stage")
    treatment_type = st.selectbox("Treatment Type", [
        "Chemotherapy", "Radiation", "Surgery", "Combination"], key="treatment_type")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Analyze Patient Data", key="predict_btn")

if predict_button:
    # Feature engineering
    delay_days = (treatment_date - diagnosis_date).days
    diag_month = diagnosis_date.month
    treat_month = treatment_date.month
    diag_year = diagnosis_date.year
    treat_year = treatment_date.year

    # Manual encoding
    data = {
        'treatment_delay_days': delay_days,
        'bmi': bmi,
        'cholesterol_level': cholesterol,
        'age': age,
        'diagnosis_month': diag_month,
        'treatment_month': treat_month,
        'diagnosis_year': diag_year,
        'treatment_year': treat_year,
        'gender_Male': 1 if gender == 'Male' else 0,
        'asthma': 1 if asthma == 'Yes' else 0,
        'smoking_status_Passive Smoker': 1 if smoking_status == 'Passive Smoker' else 0,
        'smoking_status_Former Smoker': 1 if smoking_status == 'Former Smoker' else 0,
        'treatment_type_Combination': 1 if treatment_type == 'Combination' else 0,
        'treatment_type_Radiation': 1 if treatment_type == 'Radiation' else 0,
        'cancer_stage_Stage III': 1 if cancer_stage == 'Stage III' else 0
    }

    # Fill missing one-hot fields with 0 if not selected
    for col in features:
        if col not in data:
            data[col] = 0

    input_df = pd.DataFrame([data])[features]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    # Results section


    if prediction == 1:
        st.markdown(f"""
            <div class="success-result">
                <div class="icon">✅</div>
                <strong>Positive Survival Prediction</strong><br>
                Confidence Level: {probability:.1f}%
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="medical-info">
                <strong>Clinical Interpretation:</strong> Based on the analyzed parameters, 
                the patient shows favorable indicators for survival outcomes. Continue monitoring 
                and follow recommended treatment protocols.
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
            <div class="warning-result">
                <div class="icon">⚠️</div>
                <strong>Requires Enhanced Care</strong><br>
                Risk Assessment: {100 - probability:.1f}%
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="medical-info">
                <strong>Clinical Recommendation:</strong> The analysis indicates elevated risk factors. 
                Consider additional interventions, closer monitoring, and multidisciplinary care approach.
            </div>
        """, unsafe_allow_html=True)

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Treatment Delay", f"{delay_days} days")
    with col2:
        st.metric("Patient Age", f"{age} years")
    with col3:
        st.metric("Cancer Stage", cancer_stage)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer disclaimer
st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This prediction tool is designed for clinical decision support only. 
        Results should not replace professional medical judgment. Always consult with qualified healthcare 
        professionals for comprehensive patient care and treatment decisions. This system is based on 
        statistical models and historical data patterns.
    </div>
""", unsafe_allow_html=True)

# Additional medical information
st.markdown("---")
st.markdown("""
    <div class="info-card">
        <h4>🔬 About Our Prediction Model</h4>
        <p>Our Random Forest algorithm analyzes multiple clinical variables including:</p>
        <ul>
            <li><strong>Temporal Factors:</strong> Treatment timing and diagnosis dates</li>
            <li><strong>Patient Demographics:</strong> Age, gender, and lifestyle factors</li>
            <li><strong>Clinical Parameters:</strong> BMI, cholesterol levels, and comorbidities</li>
            <li><strong>Treatment Variables:</strong> Cancer stage and treatment modality</li>
        </ul>
        <p>The model has been trained on clinical datasets and validated for accuracy in survival prediction.</p>
    </div>
""", unsafe_allow_html=True)