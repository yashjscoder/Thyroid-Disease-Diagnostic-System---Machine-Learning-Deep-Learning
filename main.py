import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import seaborn as sns
# --- PDF GENERATION FUNCTION ---
def generate_pdf(age, sex, prob, result_text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "AI THYROID DIAGNOSTIC REPORT")
    p.line(100, 745, 500, 745)
    
    # Patient Info
    p.setFont("Helvetica", 12)
    p.drawString(100, 710, f"Patient Age: {age}")
    p.drawString(100, 690, f"Patient Sex: {'Male' if sex == 1 else 'Female'}")
    
    # Diagnosis Results
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 650, "Diagnosis Results:")
    p.setFont("Helvetica", 12)
    p.drawString(100, 630, f"AI Assessment: {result_text}")
    p.drawString(100, 610, f"AI Confidence Score: {prob:.2%}")
    
    # Disclaimer
    p.setFont("Helvetica-Oblique", 10)
    p.drawString(100, 550, "Disclaimer: This is an AI-generated report for educational purposes.")
    p.drawString(100, 535, "Please consult a professional medical doctor for clinical diagnosis.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# function to plot feature importance
def plot_importance(model, input_df):
    # Get feature importance scores from XGBoost
    importance = model.get_score(importance_type='weight')
    
    # Sort and take top 10
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(sorted_importance.values()), y=list(sorted_importance.keys()), ax=ax, palette='viridis')
    plt.title("Top Factors Influencing This Diagnosis")
    plt.xlabel("Impact Score")
    return fig

# 1. Page Configuration
st.set_page_config(page_title="AI Thyroid Doctor", page_icon="⚕️", layout="wide")

# 2. Asset Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'thyroid_model.json')
scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')

@st.cache_resource
def load_assets():
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# 3. UI Header
st.title("⚕️ Virtual Thyroid Diagnostic System")
if model is None:
    st.error("⚠️ Model Files Missing in current directory!")
    st.stop()

st.divider()

# 4. Input Form
with st.form("medical_form"):
    st.subheader("Patient Clinical Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 2)], format_func=lambda x: x[0])[1]
        on_thyroxine = st.selectbox("On Thyroxine?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        sick = st.selectbox("Feeling 'Sick'?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        pregnant = st.selectbox("Pregnant?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col2:
        tsh = st.number_input("TSH Level", value=1.5, format="%.4f")
        t3 = st.number_input("T3 Level", value=2.0, format="%.4f")
        tt4 = st.number_input("TT4 Level", value=100.0, format="%.4f")
        tumor = st.selectbox("History of Tumors?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col3:
        t4u = st.number_input("T4U Level", value=1.0, format="%.4f")
        fti = st.number_input("FTI Level", value=100.0, format="%.4f")
        goitre = st.selectbox("Goitre?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        psych = st.selectbox("Psychological Symptoms?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    
    submit = st.form_submit_button("Run AI Diagnosis")

# 5. Logical Sequence: PREDICT -> DISPLAY -> PDF
if submit:
    # --- STEP A: Prepare Data ---
    data_dict = {
        'age': [age], 'sex': [sex], 'on_thyroxine': [on_thyroxine],
        'query_on_thyroxine': [0], 'on_antithyroid_medication': [0],
        'sick': [sick], 'pregnant': [pregnant], 'thyroid_surgery': [0], 
        'I131_treatment': [0], 'query_hypothyroid': [0], 'query_hyperthyroid': [0], 
        'lithium': [0], 'goitre': [goitre], 'tumor': [tumor], 'hypopituitary': [0], 
        'psych': [psych], 'TSH': [tsh], 'T3': [t3], 'TT4': [tt4], 'T4U': [t4u], 'FTI': [fti]
    }
    input_df = pd.DataFrame(data_dict)
    
    # --- STEP B: Scaling & Prediction ---
    num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    final_input = xgb.DMatrix(input_df)
    prob = model.predict(final_input)[0]  # CALCULATED HERE

    # --- STEP C: Results Display ---
    st.divider()
    if prob > 0.5:
        res_text = "POSITIVE (Risk Detected)"
        st.error(f"### Result: {res_text}")
        st.write(f"Confidence Level: **{prob:.2%}**")
    else:
        res_text = "NEGATIVE (Healthy)"
        st.success(f"### Result: {res_text}")
        st.write(f"Confidence Level: **{(1-prob):.2%}**")

     # Inside 'if submit:' block, after showing the result text
    st.subheader("📊 Diagnostic Analysis")
    st.write("This chart shows which clinical markers most influenced the AI's decision for this specific patient.")
    
    importance_fig = plot_importance(model, input_df)
    st.pyplot(importance_fig)

    # --- STEP D: Generate PDF (Now prob and res_text exist!) ---
    pdf_file = generate_pdf(age, sex, prob, res_text)
    st.download_button(
        label="📄 Download Medical Report (PDF)",
        data=pdf_file,
        file_name=f"Thyroid_Report_{age}.pdf",
        mime="application/pdf"
    )