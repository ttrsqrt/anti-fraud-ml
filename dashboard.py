import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Fraud Detection System", layout="wide")

API_URL = "http://localhost:8000/predict"

st.title("🛡️ Anti-Fraud Monitoring System")

# Sidebar for controls
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Blocking Threshold", 0.0, 1.0, 0.5, 0.01)

# Input method
input_method = st.sidebar.radio("Input Method", ["Manual Input", "Load Sample from Dataset"])

input_data = {}

if input_method == "Load Sample from Dataset":
    # Load a sample from the featured dataset
    try:
        df = pd.read_csv(r"c:\Users\User\Documents\ttrsqr\featured_dataset.csv")
        # Pick a random row or let user select by ID
        sample_id = st.sidebar.number_input("Select Row Index", 0, len(df)-1, 0)
        row = df.iloc[sample_id]
        
        # Filter columns to match what the API expects (exclude target, IDs, dates)
        # For MVP, we send everything except the non-features
        exclude = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target']
        input_data = row.drop(labels=exclude).to_dict()
        
        st.info(f"Loaded transaction for Client {row['cst_dim_id']} on {row['transdate']}")
        
        # Show ground truth
        is_fraud_truth = row['target'] == 1
        if is_fraud_truth:
            st.error("⚠️ Ground Truth: FRAUD")
        else:
            st.success("✅ Ground Truth: LEGITIMATE")
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

else:
    # Manual input (simplified for demo)
    st.warning("Manual input requires entering many features. Using defaults for demo.")
    # In a real app, we'd have a form here.
    # For now, let's just create a dummy suspicious transaction
    input_data = {
        "amount": 50000,
        "time_since_last_trans": 60, # 1 minute
        "amount_mean_30d": 5000,
        "amount_zscore_30d": 10.0,
        "device_changed": 1,
        "unique_senders_to_receiver": 50,
        # Add other necessary fields with defaults...
        # This is tricky without knowing all columns.
        # Best to use the "Load Sample" for the demo.
    }

if st.button("Analyze Transaction"):
    if not input_data:
        st.error("No data to analyze.")
    else:
        try:
            # Handle potential NaN in input_data (JSON doesn't like NaN)
            clean_data = {k: (v if pd.notna(v) else 0) for k, v in input_data.items()}
            
            response = requests.post(API_URL, json=clean_data)
            
            if response.status_code == 200:
                result = response.json()
                prob = result['fraud_probability']
                decision = "BLOCK" if prob > threshold else "ALLOW"
                
                # Dashboard Layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Risk Score")
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prob > threshold else "green"},
                            'steps': [
                                {'range': [0, threshold*100], 'color': "lightgreen"},
                                {'range': [threshold*100, 100], 'color': "salmon"}],
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    if decision == "BLOCK":
                        st.error(f"⛔ DECISION: {decision}")
                    else:
                        st.success(f"✅ DECISION: {decision}")
                
                with col2:
                    st.subheader("Analysis & Explanation")
                    
                    expl = result['explanation']
                    st.markdown(f"**LLM Summary:**\n> {expl['llm_summary']}")
                    
                    st.markdown("**Key Risk Factors:**")
                    st.text(expl['text_explanation'])
                    
                    # Feature Importance Plot (Top 3)
                    top_feats = pd.DataFrame(expl['top_features'])
                    st.bar_chart(top_feats.set_index('feature')['shap_value'])
                    
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Error: {e}. Is the API running?")

st.markdown("---")
st.caption("Fraud Detection System MVP | Powered by CatBoost & SHAP")
