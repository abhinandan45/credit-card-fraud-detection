# 💳 Credit Card Fraud Detection System

End-to-end Machine Learning system to detect fraud transactions in real-time.

## 🎯 Problem Statement
Detect fraudulent credit card transactions from highly imbalanced dataset (0.17% fraud).

## 🔧 Tech Stack
- **ML:** XGBoost, Scikit-learn, SHAP
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Deployment:** Render + Streamlit Cloud

## 📊 Model Performance
- **F1 Score:** 0.85
- **Recall:** 0.83
- **Precision:** 0.87

## 🚀 Live Demo
- **Frontend:** https://credit-card-fraud-detection-a45.streamlit.app
- **API Docs:** https://credit-card-fraud-detection-yyq3.onrender.com

## 🛠️ How to Run

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
