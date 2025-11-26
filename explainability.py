import shap
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(SCRIPT_DIR, "lgbm_model.pkl")

class FraudExplainer:
    def __init__(self, model_path=MODEL_FILE):
        print(f"Loading model from {model_path}...")
        # Load LightGBM model using joblib
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        
    def get_shap_values(self, X):
        # Calculate SHAP values for the given data
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def generate_explanation(self, row_data, shap_values, feature_names):
        # Generate a text explanation based on top SHAP features
        # row_data: pandas Series or dict of feature values
        # shap_values: numpy array of shap values for this row
        
        # Create a DataFrame of features and their impact
        impact = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'value': row_data.values
        })
        
        # Sort by absolute impact
        impact['abs_shap'] = impact['shap_value'].abs()
        impact = impact.sort_values('abs_shap', ascending=False)
        
        top_features = impact.head(3)
        
        explanation = "Suspicious Activity Detected:\n"
        reasons = []
        
        for _, row in top_features.iterrows():
            feat = row['feature']
            val = row['value']
            shap_val = row['shap_value']
            
            # Simple rule-based text generation (Mock LLM)
            if shap_val > 0:
                direction = "increases risk"
            else:
                direction = "decreases risk"
                
            reasons.append(f"- {feat} (value: {val}) {direction} (impact: {shap_val:.2f})")
            
        explanation += "\n".join(reasons)
        
        # Mock LLM call
        llm_summary = self._mock_llm_generate(reasons)
        
        return {
            'text_explanation': explanation,
            'llm_summary': llm_summary,
            'top_features': top_features.to_dict(orient='records')
        }

    def _mock_llm_generate(self, reasons):
        # In a real scenario, this would call OpenAI/Anthropic API
        prompt = f"Explain why this transaction is fraudulent based on: {reasons}"
        
        # Mock response
        return "The transaction is flagged as high risk primarily due to unusual frequency of transfers to this recipient and a high transaction amount relative to the user's history."

if __name__ == "__main__":
    # Test run
    print("Testing FraudExplainer...")
    try:
        explainer = FraudExplainer()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
