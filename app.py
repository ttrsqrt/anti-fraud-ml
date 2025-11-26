from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from explainability import FraudExplainer
import joblib

app = FastAPI(title="Fraud Detection API")

# Load resources
try:
    explainer = FraudExplainer()
    print("Model and Explainer loaded.")
except Exception as e:
    print(f"Error loading resources: {e}")
    explainer = None

class TransactionInput(BaseModel):
    # We need all features required by the model
    # For simplicity in MVP, we might accept raw fields and do feature engineering on the fly
    # OR accept pre-calculated features.
    # Given the complexity of rolling windows, a real production system would use a Feature Store.
    # For this MVP, we will assume the input contains the PRE-CALCULATED features.
    
    # List of features expected by the model (from train_model.py output)
    time_since_last_trans: float
    amount_mean_1d: float
    amount_mean_7d: float
    amount_mean_30d: float
    amount_std_30d: float
    amount_zscore_30d: float
    trans_count_1h: float
    device_changed: int
    os_changed: int
    unique_senders_to_receiver: int
    unique_receivers_from_sender: int
    
    # Original features needed for context or simple pass-through
    amount: float
    hour: int
    day_of_week: int
    day_of_month: int
    is_weekend: int
    is_night: int
    amount_log: float
    is_round_amount: int
    direction_frequency: float
    phone_model_frequency: float
    os_frequency: float
    
    # Categorical (passed as is or encoded? CatBoost handles strings/int)
    # Our model was trained with these as features?
    # Let's check the feature list from training.
    # Based on feature_engineering.py, we have many features.
    # We should probably make the input model dynamic or cover the top features.

@app.post("/predict")
def predict_fraud(transaction: dict):
    if not explainer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([transaction])
        
        # Clean feature names to match training (remove special chars)
        import re
        df.columns = [re.sub(r'[^\w]', '_', col) for col in df.columns]
        
        # Ensure categorical columns are category type
        cat_features = ['last_phone_model', 'last_os'] # Add other cat features if any
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Get prediction
        prob = explainer.model.predict_proba(df)[0][1]
        prediction = int(explainer.model.predict(df)[0])
        
        # Get explanation
        shap_values = explainer.get_shap_values(df)
        
        # Handle SHAP output variations (List for multiclass/some binary, Array for others)
        import numpy as np
        if isinstance(shap_values, list):
            # For binary classification, we usually want the positive class (index 1)
            # If only 1 output, take index 0
            if len(shap_values) > 1:
                sv = shap_values[1]
            else:
                sv = shap_values[0]
        else:
            sv = shap_values
            
        # sv is now likely (n_samples, n_features)
        # We take the first sample
        sv = sv[0]
        
        # Ensure it is 1D array
        sv = np.array(sv).flatten()
        
        feature_names = df.columns.tolist()
        
        # Ensure feature_names and sv have same length
        # SHAP sometimes adds a bias column? Usually not in shap_values() unless requested.
        # But if lengths differ, we truncate or pad? 
        # Better to just pass what we have and let generate_explanation handle or error with more info.
        
        explanation = explainer.generate_explanation(df.iloc[0], sv, feature_names, is_fraud=bool(prediction))
        
        return {
            "fraud_probability": float(prob),
            "is_fraud": bool(prediction),
            "decision": "BLOCK" if prob > 0.5 else "ALLOW", # Threshold can be tuned
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
