import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 1. Page Configuration
st.set_page_config(page_title="AI Thyroid Doctor", page_icon="⚕️", layout="centered")

# 2. Load the Saved Model and Scaler
@st.cache_resource
def load_assets():
    model = xgb.Booster()
    model.load_model('thyroid_model.json')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ Model files not found! Make sure 'thyroid_model.json' and 'scaler.joblib' are in the same folder.")

# 3. UI Header
st.title("⚕️ Thyroid Disease Detection System")
st.write("Enter patient clinical data below for an AI-powered diagnosis.")
st.divider()

# 4. Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 2)], format_func=lambda x: x[0])[1]
        tsh = st.number_input("TSH Level", value=1.5)
        t3 = st.number_input("T3 Level", value=2.0)
    
    with col2:
        tt4 = st.number_input("TT4 Level", value=100.0)
        t4u = st.number_input("T4U Level", value=1.0)
        fti = st.number_input("FTI Level", value=100.0)
        on_thyroxine = st.selectbox("On Thyroxine?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    submit = st.form_submit_button("Generate Diagnosis")

# 5. Prediction Logic
if submit:
    # Prepare the raw data (Match the exact order of your training columns!)
    # Note: You might need to add other 't/f' features as 0s if your model was trained on them
    input_data = pd.DataFrame([[age, sex, tsh, t3, tt4, t4u, fti, on_thyroxine]], 
                              columns=['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'on_thyroxine'])
    
    # Scale the numerical parts
    numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)[0]

    # 6. Display Results
    st.subheader("Results:")
    if prediction > 0.5:
        st.error(f"🚨 **High Risk Detected.** (Confidence: {prediction:.2%})")
        st.write("The model suggests clinical signs of Thyroid disease. Please consult a specialist.")
    else:
        st.success(f"✅ **Healthy / Low Risk.** (Confidence: {(1-prediction):.2%})")
        st.write("Thyroid levels appear to be within the normal range based on the data provided.")