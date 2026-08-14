import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# ==================== CONFIG ====================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==================== SIDEBAR ====================
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Info"]
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("### AI-Powered Real-time Fraud Detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "99.9%", "XGBoost")
    with col2:
        st.metric("F1 Score", "0.85", "Best Model")
    with col3:
        st.metric("Recall", "83%", "Fraud Detection")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 About This Project
    
    This is an end-to-end **Machine Learning System** that detects 
    credit card fraud in real-time using **XGBoost** algorithm.
    
    ### 🔧 Tech Stack
    - **Backend:** FastAPI (REST API)
    - **Frontend:** Streamlit
    - **ML Model:** XGBoost with SHAP Explainability
    - **Techniques:** SMOTE, Class Weighting, Cross-Validation
    
    ### 📊 Dataset
    - **284,807** total transactions
    - **492** fraud cases (0.17%)
    - **Highly Imbalanced** - Real world challenge
    
    ### ✅ Features
    - Real-time fraud detection
    - Batch processing (CSV upload)
    - Confidence scores
    - Risk level assessment
    """)

# ==================== SINGLE PREDICTION ====================
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Transaction Prediction")
    st.markdown("Test a transaction to check if it's fraud")
    
    # Get Samples
    try:
        response = requests.get(f"{API_URL}/samples")
        samples = response.json()['samples']
        
        # Sample Selection
        st.markdown("### 📋 Select a Sample Transaction")
        
        sample_options = {
            f"{s['id']} - {s['type']}": s for s in samples
        }
        
        selected = st.selectbox(
            "Choose:",
            list(sample_options.keys())
        )
        
        selected_sample = sample_options[selected]
        
        # Show Data
        with st.expander("📄 View Transaction Data"):
            st.json(selected_sample['data'])
        
        # Predict Button
        if st.button("🔮 Predict Fraud", type="primary"):
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=selected_sample['data']
                )
                result = response.json()
                
                # Show Results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result['is_fraud']:
                        st.error(f"🚨 {result['prediction']}")
                    else:
                        st.success(f"✅ {result['prediction']}")
                
                with col2:
                    fraud_prob = result['confidence']['fraud'] * 100
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_prob:.2f}%"
                    )
                
                with col3:
                    risk = result['risk_level']
                    if risk == "HIGH RISK":
                        st.error(f"⚠️ {risk}")
                    elif risk == "MEDIUM RISK":
                        st.warning(f"⚠️ {risk}")
                    elif risk == "LOW RISK":
                        st.info(f"ℹ️ {risk}")
                    else:
                        st.success(f"✅ {risk}")
                
                # Confidence Chart
                st.markdown("### 📈 Confidence Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Normal', 'Fraud'],
                        y=[result['confidence']['normal']*100, 
                           result['confidence']['fraud']*100],
                        marker_color=['green', 'red']
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Confidence (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Make sure FastAPI server is running on port 8000")

# ==================== BATCH PREDICTION ====================
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Prediction")
    st.markdown("Upload CSV file with multiple transactions")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Preview
        df = pd.read_csv(uploaded_file)
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head())
        
        st.info(f"Total Transactions: {len(df)}")
        
        if st.button("🚀 Analyze All", type="primary"):
            with st.spinner("Processing..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                files = {"file": uploaded_file}
                response = requests.post(
                    f"{API_URL}/predict_batch",
                    files=files
                )
                
                result = response.json()
                summary = result['summary']
                predictions = result['predictions']
                
                # Summary
                st.markdown("### 📊 Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", summary['total_transactions'])
                with col2:
                    st.metric("Fraud", summary['fraud_detected'], 
                             delta_color="inverse")
                with col3:
                    st.metric("Normal", summary['normal_transactions'])
                with col4:
                    st.metric("Fraud %", f"{summary['fraud_percentage']}%")
                
                # Pie Chart
                st.markdown("### 🥧 Distribution")
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Normal', 'Fraud'],
                        values=[summary['normal_transactions'],
                               summary['fraud_detected']],
                        marker_colors=['green', 'red']
                    )
                ])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results Table
                st.markdown("### 📋 Detailed Results")
                
                results_df = pd.DataFrame(predictions)
                results_df['fraud_probability'] = results_df['fraud_probability'].apply(
                    lambda x: f"{x*100:.2f}%"
                )
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download
                csv = pd.DataFrame(predictions).to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )

# ==================== MODEL INFO ====================
elif page == "📈 Model Info":
    st.title("📈 Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model_info")
        info = response.json()
        
        st.markdown(f"### 🤖 {info['model_type']}")
        
        # Training Details
        st.markdown("### 📊 Training Details")
        details = info['training_details']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset", "Credit Card")
        with col2:
            st.metric("Total Samples", f"{details['total_samples']:,}")
        with col3:
            st.metric("Fraud Cases", details['fraud_samples'])
        
        # Performance
        st.markdown("### 🎯 Model Performance")
        perf = info['performance']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1 Score", perf['f1_score'])
        with col2:
            st.metric("Recall", perf['recall'])
        with col3:
            st.metric("Precision", perf['precision'])
        
        # Confusion Info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("False Negatives", perf['false_negatives'])
        with col2:
            st.metric("False Positives", perf['false_positives'])
        
        # Techniques
        st.markdown("### 🛠️ Techniques Used")
        for tech in info['techniques_used']:
            st.markdown(f"- ✅ {tech}")
    
    except Exception as e:
        st.error(f"Error: {e}")