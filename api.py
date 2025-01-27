from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained models
heart_model = joblib.load('heart.pkl')
diabetes_model = joblib.load('diabetes.pkl')

# Initialize FastAPI app
app = FastAPI()

# Input schema for heart disease prediction
class HeartInput(BaseModel):
    age: int
    sex: str
    chest_pain: str
    resting_bp: int
    serum_chol: int
    fasting_bs: str
    resting_ecg: str
    max_hr: int
    exercise_angina: str
    st_depression: float
    st_slope: str
    ca: int
    thal: str

# Input schema for diabetes prediction
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int

# Route to check API health
@app.get("/health")
async def health_check():
    return {"status": "API is running successfully"}

# Endpoint for heart disease prediction
@app.post("/predict/heart")
async def predict_heart(input_data: HeartInput):
    try:
        # Map categorical inputs to numeric values
        sex_code = 1 if input_data.sex.lower() == 'male' else 0
        chest_pain_types = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
        fasting_bs_code = 1 if input_data.fasting_bs.lower() == 'yes' else 0
        resting_ecg_types = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
        exercise_angina_code = 1 if input_data.exercise_angina.lower() == 'yes' else 0
        st_slope_types = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        thal_types = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2, 'Unknown': 3}
        
        chest_pain_code = chest_pain_types.get(input_data.chest_pain, -1)
        resting_ecg_code = resting_ecg_types.get(input_data.resting_ecg, -1)
        st_slope_code = st_slope_types.get(input_data.st_slope, -1)
        thal_code = thal_types.get(input_data.thal, -1)

        # Create input array for the model
        model_input = np.array([
            input_data.age, 
            sex_code, 
            chest_pain_code, 
            input_data.resting_bp, 
            input_data.serum_chol, 
            fasting_bs_code, 
            resting_ecg_code, 
            input_data.max_hr, 
            exercise_angina_code, 
            input_data.st_depression, 
            st_slope_code, 
            input_data.ca, 
            thal_code
        ]).reshape(1, -1)
        
        prediction = heart_model.predict(model_input)
        result = "Heart disease detected" if prediction[0] == 1 else "No heart disease detected"
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in heart disease prediction: {str(e)}")

# Endpoint for diabetes prediction
@app.post("/predict/diabetes")
async def predict_diabetes(input_data: DiabetesInput):
    try:
        # Create input array for the model
        model_input = np.array([
            input_data.pregnancies, 
            input_data.glucose, 
            input_data.blood_pressure, 
            input_data.skin_thickness, 
            input_data.insulin, 
            input_data.bmi, 
            input_data.diabetes_pedigree, 
            input_data.age
        ]).reshape(1, -1)
        
        prediction = diabetes_model.predict(model_input)
        result = "Diabetes detected" if prediction[0] == 1 else "No diabetes detected"
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in diabetes prediction: {str(e)}")


