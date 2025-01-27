import streamlit as st
import numpy as np
import joblib
import pandas as pd
from fpdf import FPDF
import base64
from datetime import datetime

# Load pre-trained models
heart_model = joblib.load('heart.pkl')
diabetes_model = joblib.load('diabetes.pkl')

# Function to take inputs for heart disease prediction
def heart_input_features():
    name = st.text_input('Name')
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', ('Male', 'Female'))
    
    chest_pain_types = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    chest_pain = st.selectbox('Chest Pain Type', list(chest_pain_types.values()))
    chest_pain_code = list(chest_pain_types.keys())[list(chest_pain_types.values()).index(chest_pain)]
    
    resting_bp = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=50, max_value=200, value=120)
    serum_chol = st.number_input('Serum Cholesterol (in mg/dL)', min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ('No', 'Yes'))
    fasting_bs_code = 1 if fasting_bs == 'Yes' else 0
    
    resting_ecg_types = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
    resting_ecg = st.selectbox('Resting ECG', list(resting_ecg_types.values()))
    resting_ecg_code = list(resting_ecg_types.keys())[list(resting_ecg_types.values()).index(resting_ecg)]
    
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    
    exercise_angina_options = {0: 'No', 1: 'Yes'}
    exercise_angina = st.selectbox('Exercise Induced Angina', list(exercise_angina_options.values()))
    exercise_angina_code = list(exercise_angina_options.keys())[list(exercise_angina_options.values()).index(exercise_angina)]
    
    st_depression = st.number_input('ST Depression (exercise vs rest)', min_value=0.0, max_value=10.0, value=1.0)
    
    st_slope_types = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', list(st_slope_types.values()))
    st_slope_code = list(st_slope_types.keys())[list(st_slope_types.values()).index(st_slope)]
    
    ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    
    thalassemia_types = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown'}
    thal = st.selectbox('Thalassemia', list(thalassemia_types.values()))
    thal_code = list(thalassemia_types.keys())[list(thalassemia_types.values()).index(thal)]
    
    user_data = np.array([age, 1 if sex == 'Male' else 0, chest_pain_code, resting_bp, serum_chol, fasting_bs_code,
                          resting_ecg_code, max_hr, exercise_angina_code, st_depression, st_slope_code, ca, thal_code]).reshape(1, -1)
    
    user_info = {
        'Name': name, 
        'Age': age, 
        'Sex': sex, 
        'Chest Pain Type': chest_pain, 
        'Resting BP': resting_bp,
        'Serum Cholesterol': serum_chol, 
        'Fasting BS > 120 mg/dL': fasting_bs, 
        'Resting ECG': resting_ecg,
        'Max HR': max_hr, 
        'Exercise Angina': exercise_angina, 
        'ST Depression': st_depression,
        'ST Slope': st_slope, 
        'Number of Vessels': ca, 
        'Thalassemia': thal
    }
    
    return user_data, user_info

# Function to take inputs for diabetes prediction
def diabetes_input_features():
    name = st.text_input('Name')
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin Level (IU/mL)', min_value=0, max_value=900, value=30)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=60.0, value=25.0)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    
    user_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
    
    user_info = {
        'Name': name, 'Pregnancies': pregnancies, 'Glucose Level': glucose, 'Blood Pressure': blood_pressure,
        'Skin Thickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'Diabetes Pedigree Function': diabetes_pedigree, 'Age': age
    }
    return user_data, user_info

# Function to generate a PDF report
def generate_pdf(user_info, result, condition_color):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Medical Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    for key, value in user_info.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.set_text_color(*condition_color)
    pdf.cell(200, 10, txt=f"Diagnosis: {result}", ln=True)
    pdf.set_text_color(0, 0, 0)
    
    return pdf

# Streamlit UI setup
st.title('Medical Test Prediction System')

# Select between Heart Disease and Diabetes Prediction
option = st.radio("Choose a test:", ('Heart Disease', 'Diabetes'))

if option == 'Heart Disease':
    input_data, user_info = heart_input_features()
    model = heart_model
    positive_diagnosis = 'The person has heart disease.'
    negative_diagnosis = 'The person does not have heart disease.'
else:
    input_data, user_info = diabetes_input_features()
    model = diabetes_model
    positive_diagnosis = 'The person has diabetes.'
    negative_diagnosis = 'The person does not have diabetes.'

if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        result = negative_diagnosis
        st.success(result)
        condition_color = (0, 128, 0)  # Green for no disease
    else:
        result = positive_diagnosis
        st.warning(result)
        condition_color = (255, 0, 0)  # Red for disease

    st.subheader('Entered Information:')
    st.write(pd.DataFrame([user_info]))

    pdf = generate_pdf(user_info, result, condition_color)
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{user_info["Name"]}_{option}_Report.pdf">Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)
