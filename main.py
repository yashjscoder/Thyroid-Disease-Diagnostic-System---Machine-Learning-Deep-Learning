import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Thyroid Doctor", page_icon="⚕️", layout="wide")

# 2. Path Debugging & Asset Loading
# This tells the app to look in the same folder as the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'thyroid_model.json')
scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')

@st.cache_resource
def load_assets():
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    # Load XGBoost Model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# 3. UI Header
st.title("⚕️ Virtual Thyroid Diagnostic System")
st.write("Professional AI tool for thyroid risk assessment based on clinical markers.")

if model is None:
    st.error(f"⚠️ **Model Files Missing!**")
    st.info(f"The app is looking in: `{BASE_DIR}`")
    st.warning("Please ensure 'thyroid_model.json' and 'scaler.joblib' are in that exact folder.")
    st.stop()

st.divider()

# 4. Input Form
with st.form("medical_form"):
    st.subheader("Patient Clinical Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 2)], format_func=lambda x: x[0])[1]
        on_thyroxine = st.selectbox("Currently on Thyroxine?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        sick = st.selectbox("Does the patient feel 'sick'?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        pregnant = st.selectbox("Is the patient pregnant?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    with col2:
        tsh = st.number_input("TSH Level (Measured)", value=1.5, format="%.4f")
        t3 = st.number_input("T3 Level (Measured)", value=2.0, format="%.4f")
        tt4 = st.number_input("TT4 Level (Measured)", value=100.0, format="%.4f")
        tumor = st.selectbox("History of Tumors?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    with col3:
        t4u = st.number_input("T4U Level (Measured)", value=1.0, format="%.4f")
        fti = st.number_input("FTI Level (Measured)", value=100.0, format="%.4f")
        goitre = st.selectbox("Goitre Present?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        psych = st.selectbox("Psychological Symptoms?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    submit = st.form_submit_button("Run AI Diagnosis")

# 5. Prediction Logic
if submit:
    # We must provide ALL 20 columns in the exact order the model was trained
    # Defaulting rare medical conditions to 0 (False) to match the notebook features
    data_dict = {
        'age': [age], 
        'sex': [sex], 
        'on_thyroxine': [on_thyroxine],
        'query_on_thyroxine': [0], 
        'on_antithyroid_medication': [0],
        'sick': [sick],  # <--- Add this line here!
        'pregnant': [pregnant], 
        'thyroid_surgery': [0], 
        'I131_treatment': [0],
        'query_hypothyroid': [0], 
        'query_hyperthyroid': [0], 
        'lithium': [0],
        'goitre': [goitre], 
        'tumor': [tumor], 
        'hypopituitary': [0], 
        'psych': [psych],
        'TSH': [tsh], 
        'T3': [t3], 
        'TT4': [tt4], 
        'T4U': [t4u], 
        'FTI': [fti]
    }
    
    input_df = pd.DataFrame(data_dict)
    
    # Scale numerical columns
    num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # XGBoost DMatrix
    final_input = xgb.DMatrix(input_df)
    prob = model.predict(final_input)[0]

    # 6. Results Display
    st.divider()
    if prob > 0.5:
        st.error(f"### Result: POSITIVE for Thyroid Disease")
        st.write(f"Confidence Level: **{prob:.2%}**")
        st.warning("Recommendation: Patient should undergo clinical blood work (Full Thyroid Panel) and consult an endocrinologist.")
    else:
        st.success(f"### Result: NEGATIVE (Healthy)")
        st.write(f"Confidence Level: **{(1-prob):.2%}**")
        st.info("The AI suggests thyroid function is within the normal range based on the provided inputs.")