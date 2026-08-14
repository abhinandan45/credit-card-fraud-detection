from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, List


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML API to Detect Fraud Transactions",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load('models/fraud_model_xgb_final.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('data/samples.json', 'r') as f:
    samples_data = json.load(f)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is Running!",
        "status": "healthy",
        "version": "1.0"
    }


@app.get("/samples")
def get_samples():
    return {
        "total": len(samples_data),
        "samples": samples_data
    }


@app.post("/predict")
def predict_transaction(transaction: Dict):
    try:
        df = pd.DataFrame([transaction])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        result = {
            "prediction": "Fraud" if prediction == 1 else "Normal",
            "is_fraud": bool(prediction),
            "confidence": {
                "normal": float(probability[0]),
                "fraud": float(probability[1])
            },
            "risk_level": get_risk_level(probability[1])
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(contents))

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        results = []

        for i, pred in enumerate(predictions):
            results.append({
                "transaction_id": i + 1,
                "prediction": "Fraud" if pred == 1 else "Normal",
                "is_fraud": bool(pred),
                "fraud_probability": float(probabilities[i][1])
            })

        total = len(predictions)
        fraud_count = int(sum(predictions))

        return {
            "summary": {
                "total_transactions": total,
                "fraud_detected": fraud_count,
                "normal_transactions": total - fraud_count,
                "fraud_percentage": round((fraud_count/total)*100, 2)
            },
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model_info")
def model_info():
    """Model Ki Information Return Karo"""
    return {
        "model_type": "XGBoost Classifier",
        "training_details": {
            "dataset": "Credit Card Fraud Detection",
            "total_samples": 284807,
            "fraud_samples": 492,
            "imbalance_ratio": "0.17%"
        },
        "performance": {
            "f1_score": 0.85,
            "recall": 0.88,
            "precision": 0.69,
            "false_negatives": 12,
            "false_positives": 38
        },
        "techniques_used": [
            "scale_pos_weight for imbalance",
            "SHAP for explainability",
            "Multiple models comparison"
        ]
    }


# ==================== HELPER FUNCTIONS ====================

def get_risk_level(fraud_prob):
    """Fraud Probability Se Risk Level Nikaalo"""
    if fraud_prob >= 0.8:
        return "HIGH RISK"
    elif fraud_prob >= 0.5:
        return "MEDIUM RISK"
    elif fraud_prob >= 0.2:
        return "LOW RISK"
    else:
        return "SAFE"


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
