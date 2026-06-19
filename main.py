# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import FileResponse

app = FastAPI(title="HealPRO AI Diagnostic Assistant Engine")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning models
models = {}
for name in ["diabetes", "heart", "hypertension", "kidney"]:
    try:
        models[name] = joblib.load(f"{name}_model.pkl")
    except Exception as e:
        models[name] = None
        print(f"Warning: {name} model file not loaded. Error: {e}")

# Data Schema for prediction endpoints
class DiagnosticData(BaseModel):
    age: float
    bmi: float
    blood_pressure: float
    glucose_or_cholesterol: float
    family_history: int  # 0 or 1

# Chat Input Schema
class ChatInput(BaseModel):
    message: str

# Report Request Schema
class ReportRequest(BaseModel):
    age: float
    bmi: float
    blood_pressure: float
    glucose_or_cholesterol: float
    family_history: int
    disease: str
    risk_percentage: float

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/predict/{disease}")
def predict_disease(disease: str, data: DiagnosticData):
    if disease not in models:
        raise HTTPException(status_code=400, detail="Invalid disease screening condition.")
        
    model = models[disease]
    if not model:
        raise HTTPException(status_code=500, detail=f"{disease.capitalize()} model not trained or loaded.")
    
    input_features = np.array([[data.age, data.bmi, data.blood_pressure, data.glucose_or_cholesterol, data.family_history]])
    probabilities = model.predict_proba(input_features)[0]
    risk_percentage = int(probabilities[1] * 100)
    
    # Custom health recommendations based on risk percentage & disease type
    recommendation = ""
    if disease == "diabetes":
        if risk_percentage > 65:
            recommendation = "High risk detected. Consult an endocrinologist for an HbA1c screening. Limit high-glycemic carbohydrates and monitor blood sugar regularly."
        elif risk_percentage >= 35:
            recommendation = "Moderate risk detected. Consider dietary modifications, reducing refined sugars, and increasing physical activity. Monitor fasting glucose levels yearly."
        else:
            recommendation = "Low risk. Maintain healthy dietary habits and a balanced lifestyle. Continue routine general health screenings."
    elif disease == "heart":
        if risk_percentage > 65:
            recommendation = "High risk detected. Consult a cardiologist immediately for a cardiovascular assessment (ECG/stress test). Monitor BP and cholesterol closely."
        elif risk_percentage >= 35:
            recommendation = "Moderate risk. Engage in moderate cardio exercise weekly, reduce saturated fats, and schedule a lipid panel check."
        else:
            recommendation = "Low risk. Maintain an active lifestyle and a heart-healthy diet rich in fiber and omega-3 fatty acids."
    elif disease == "hypertension":
        if risk_percentage > 65:
            recommendation = "High risk detected. Consult a physician. Reduce sodium intake (<1500mg/day) and monitor blood pressure twice daily."
        elif risk_percentage >= 35:
            recommendation = "Moderate risk. Adopt the DASH diet (high in fruits/vegetables, low in sodium) and engage in regular aerobic exercise."
        else:
            recommendation = "Low risk. Keep sodium intake moderate and maintain an active lifestyle. Monitor your blood pressure periodically."
    elif disease == "kidney":
        if risk_percentage > 65:
            recommendation = "High risk detected. Consult a nephrologist for kidney function testing (eGFR and albuminuria) and control blood pressure tightly."
        elif risk_percentage >= 35:
            recommendation = "Moderate risk. Avoid frequent use of NSAID pain relievers (like ibuprofen). Stay well hydrated and check renal markers periodically."
        else:
            recommendation = "Low risk. Stay hydrated, maintain a balanced diet, and keep blood pressure in check to protect kidney function."

    return {
        "condition": disease.capitalize() if disease != "diabetes" else "Diabetes Mellitus",
        "risk_percentage": risk_percentage,
        "recommendation": recommendation
    }

@app.post("/report")
def generate_report(data: ReportRequest):
    risk = data.risk_percentage
    disease = data.disease

    # Validate disease
    valid_diseases = ["diabetes", "heart", "hypertension", "kidney"]
    if disease not in valid_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease type for report generation.")

    # Risk level + next checkup
    if risk >= 65:
        risk_level = "High"
        next_checkup = "Within 7 days"
    elif risk >= 35:
        risk_level = "Moderate"
        next_checkup = "Within 30 days"
    else:
        risk_level = "Low"
        next_checkup = "Within 6 months"

    # Possible causes
    causes = []
    if data.bmi > 30:
        causes.append("Obesity (BMI > 30)")
    elif data.bmi > 27:
        causes.append("Overweight (BMI 27–30)")
    if data.family_history:
        causes.append("Genetic / family history of condition")
    if data.blood_pressure > 140:
        causes.append("Stage 2 hypertension (BP > 140 mmHg)")
    elif data.blood_pressure > 130:
        causes.append("Elevated blood pressure (BP 130–140 mmHg)")
    if disease == "diabetes" and data.glucose_or_cholesterol > 125:
        causes.append("High fasting glucose (> 125 mg/dL — pre-diabetic range)")
    elif disease == "diabetes" and data.glucose_or_cholesterol > 100:
        causes.append("Borderline fasting glucose (100–125 mg/dL)")
    elif disease == "heart" and data.glucose_or_cholesterol > 239:
        causes.append("High LDL cholesterol (> 239 mg/dL)")
    elif disease == "heart" and data.glucose_or_cholesterol > 199:
        causes.append("Borderline high cholesterol (200–239 mg/dL)")
    elif disease == "hypertension" and data.glucose_or_cholesterol > 200:
        causes.append("Elevated cholesterol contributing to vascular resistance")
    elif disease == "kidney" and data.glucose_or_cholesterol > 150:
        causes.append("Elevated metabolic markers (glucose/cholesterol)")
    if data.age > 55:
        causes.append("Advanced age (> 55 years) — increased baseline risk")
    elif data.age > 40:
        causes.append("Age over 40 — moderate baseline risk factor")
    if not causes:
        causes.append("No single dominant risk factor; combination of mild indicators")

    # Disease-specific recommendations and lifestyle
    if disease == "diabetes":
        recommendations = [
            "Consult an Endocrinologist for HbA1c and oral glucose tolerance testing",
            "Reduce intake of refined sugars and high-glycaemic carbohydrates",
            "Maintain fasting blood glucose below 100 mg/dL through diet and exercise",
            "Perform at least 150 minutes of moderate aerobic activity per week",
        ]
        lifestyle = [
            "Switch to a low-GI diet (whole grains, legumes, leafy greens)",
            "Maintain consistent sleep schedule — poor sleep raises blood sugar",
            "Avoid sugary beverages; replace with water or unsweetened drinks",
        ]
    elif disease == "heart":
        recommendations = [
            "Consult a Cardiologist for ECG, lipid panel, and cardiovascular risk scoring",
            "Reduce saturated fat, trans fat, and dietary sodium intake",
            "Engage in moderate-intensity cardio exercise 5 days per week",
            "Monitor blood pressure and cholesterol at least every 3 months",
        ]
        lifestyle = [
            "Follow a Mediterranean or DASH diet for cardiovascular protection",
            "Quit smoking and limit alcohol to reduce vascular inflammation",
            "Manage chronic stress through mindfulness or structured relaxation",
        ]
    elif disease == "hypertension":
        recommendations = [
            "Consult a physician for ambulatory blood pressure monitoring",
            "Reduce daily sodium intake below 1500 mg",
            "Monitor blood pressure twice daily and maintain a log",
            "Avoid NSAIDs (e.g. ibuprofen) which can raise blood pressure",
        ]
        lifestyle = [
            "Follow the DASH diet — high in potassium, magnesium, and fibre",
            "Practice daily aerobic activity (walking, swimming) for at least 30 minutes",
            "Limit caffeine and reduce chronic stress triggers",
        ]
    elif disease == "kidney":
        recommendations = [
            "Consult a Nephrologist for eGFR and urinary albumin-creatinine ratio testing",
            "Control blood pressure tightly (target < 130/80 mmHg)",
            "Avoid frequent use of NSAID analgesics (e.g. ibuprofen, naproxen)",
            "Monitor creatinine and electrolyte levels every 3–6 months",
        ]
        lifestyle = [
            "Stay well hydrated (2–3 litres of water daily unless advised otherwise)",
            "Limit high-protein and high-phosphorus foods to reduce kidney load",
            "Avoid contrast dye procedures without nephrologist clearance",
        ]
    else:
        recommendations = ["Consult a qualified healthcare professional for further evaluation."]
        lifestyle = ["Maintain a balanced diet and regular physical activity."]

    # One-line summary
    disease_label = {
        "diabetes": "Diabetes Mellitus",
        "heart": "Heart Disease",
        "hypertension": "Hypertension",
        "kidney": "Kidney Disease",
    }.get(disease, disease.capitalize())

    summary = (
        f"Patient (Age {int(data.age)}, BMI {data.bmi:.1f}) presents with "
        f"{risk_level.lower()} risk ({int(risk)}%) for {disease_label} "
        f"based on submitted clinical indicators."
    )

    return {
        "risk_level": risk_level,
        "possible_causes": causes,
        "recommendations": recommendations,
        "lifestyle": lifestyle,
        "next_checkup": next_checkup,
        "summary": summary,
    }


# Mock Conversational AI symptom parsing
import re

@app.post("/chat")
def handle_chat(payload: ChatInput):
    msg = payload.message.lower()
    
    # Extract metrics using regex
    extracted = {}
    
    # Age: e.g. "45 years old", "age 45", "i am 45"
    age_match = re.search(r'\b(age\s*[:=]?\s*|i am\s*|i\'m\s*)?(\d{1,3})\s*(years|yrs|yo)?\b', msg)
    if age_match:
        val = int(age_match.group(2))
        if 10 < val < 110:
            extracted["age"] = val
            
    # BMI: e.g. "bmi of 24.5", "bmi is 25", "bmi 28"
    bmi_match = re.search(r'\bbmi\s*(of|is|:|=)?\s*(\d{1,2}(?:\.\d)?)\b', msg)
    if bmi_match:
        extracted["bmi"] = float(bmi_match.group(2))
        
    # BP: e.g. "bp of 120", "bp is 130", "blood pressure 140", "bp 140/90" (take systolic)
    bp_match = re.search(r'\b(bp|blood\s*pressure)\s*(of|is|:|=)?\s*(\d{2,3})(\s*/\s*\d{2,3})?\b', msg)
    if bp_match:
        extracted["blood_pressure"] = int(bp_match.group(3))
        
    # Glucose or Cholesterol: e.g. "glucose is 110", "cholesterol 190", "sugar 120", "lab 150"
    lab_match = re.search(r'\b(glucose|cholesterol|sugar|lab|value)\s*(is|of|:|=)?\s*(\d{2,3})\b', msg)
    if lab_match:
        extracted["glucose_or_cholesterol"] = int(lab_match.group(3))
        
    # Family history: e.g. "family history yes", "family history of heart disease", "father had diabetes"
    if any(phrase in msg for phrase in ["family history", "genetic", "hereditary", "parents", "father", "mother"]):
        if any(neg in msg for neg in ["no", "none", "don't", "dont", "deny"]):
            extracted["family_history"] = 0
        else:
            extracted["family_history"] = 1

    # Symptom categorisation & responsive text
    response_text = ""
    category = "General"
    suggested_disease = ""
    
    # Check for Diabetes symptoms
    has_diabetes_sym = any(word in msg for word in ["fatigue", "thirsty", "thirst", "urination", "polyuria", "sugar", "diabetes", "diabetic"])
    # Check for Heart symptoms
    has_heart_sym = any(word in msg for word in ["chest pain", "breathless", "heart", "palpitations", "angina", "shortness of breath"])
    # Check for Hypertension symptoms
    has_hyper_sym = any(word in msg for word in ["headache", "dizzy", "dizziness", "blurred vision", "nosebleed", "hypertension", "bp", "pressure"])
    # Check for Kidney symptoms
    has_kidney_sym = any(word in msg for word in ["swelling", "edema", "foamy urine", "nausea", "kidney", "renal", "flank pain"])
    
    if has_heart_sym:
        response_text = "I've flagged possible cardiovascular symptoms (e.g., chest discomfort, breathing changes). I recommend running a Heart Disease risk assessment."
        category = "Heart Disease"
        suggested_disease = "heart"
    elif has_diabetes_sym:
        response_text = "I've noted symptoms that can relate to blood sugar levels (e.g., thirst, fatigue, frequent urination). Let's check your Diabetes Mellitus risk."
        category = "Diabetes"
        suggested_disease = "diabetes"
    elif has_hyper_sym:
        response_text = "Symptoms like headaches, dizziness, or chest pressure might be linked to blood pressure. I suggest assessing your Hypertension risk."
        category = "Hypertension"
        suggested_disease = "hypertension"
    elif has_kidney_sym:
        response_text = "Noted markers like localized swelling (edema) or urine changes, which can indicate renal load. Running a Kidney Disease assessment is advised."
        category = "Kidney Disease"
        suggested_disease = "kidney"
    else:
        response_text = "I am ready to help. Describe symptoms like chest pain, fatigue, headaches, or swelling, or give me metrics (e.g., 'I am 45, BMI 28') to auto-populate the screening tool."
        category = "General"
        
    # Build feedback prompt on extracted data
    if extracted:
        items_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in extracted.items()])
        response_text += f"\n\nI parsed the following metrics from your message: **{items_str}**. Click the button below to apply them to the ML Risk Estimator form."
        
    return {
        "response": response_text,
        "category": category,
        "suggested_disease": suggested_disease,
        "extracted_data": extracted
    }
