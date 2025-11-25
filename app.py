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
        
        # Ensure correct column order/presence (CatBoost is sensitive to column order if not using Pool with feature names, 
        # but we used a dataframe for training so it should map by name if we are careful, 
        # or we just pass the dict if CatBoost supports it. Safest is DataFrame with correct columns.)
        
        # Get prediction
        prob = explainer.model.predict_proba(df)[0][1]
        prediction = int(explainer.model.predict(df)[0])
        
        # Get explanation
        shap_values = explainer.get_shap_values(df)
        # shap_values is (1, n_features)
        
        feature_names = df.columns.tolist()
        explanation = explainer.generate_explanation(df.iloc[0], shap_values[0], feature_names)
        
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
