from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_PATH / "preprocessor.pkl"
MODEL_PATH = MODELS_PATH / "best_model.pkl"

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

preprocessor = None
model = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str 
@app.on_event("startup")
def load_artifacts():
    global preprocessor, model
    try:
        logging.info("Loading model artifacts...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        logging.info("Model artifacts loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load artifacts: {e}")
        raise RuntimeError("Could not load model artifacts")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    global preprocessor, model
    
    try:
        data = customer.dict()
        df = pd.DataFrame([data])
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        X_processed = preprocessor.transform(df)
        
        churn_prob = model.predict_proba(X_processed)[0][1]
        prediction = int(churn_prob > 0.5)
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": float(churn_prob)
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
