import shap
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
import warnings
import os
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Загружаем переменные окружения из .env файла
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

MODEL_FILE = os.path.join(SCRIPT_DIR, "lgbm_model.pkl")

class FraudExplainer:
    def __init__(self, model_path=MODEL_FILE):
        print(f"Loading model from {model_path}...")
        # Load LightGBM model using joblib
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'sk-your-actual-key-here':
            print("WARNING: OpenAI API key not set. LLM explanations will use fallback mode.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                print("OpenAI client initialized successfully.")
            except Exception as e:
                print(f"WARNING: Failed to initialize OpenAI client: {e}")
                print("LLM explanations will use fallback mode.")
                self.client = None
        
    def get_shap_values(self, X):
        # Calculate SHAP values for the given data
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def generate_explanation(self, row_data, shap_values, feature_names, is_fraud=True):
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
        
        if is_fraud:
            explanation = "Suspicious Activity Detected:\n"
        else:
            explanation = "Legitimate Activity Detected:\n"
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
        
        # Real LLM call (with fallback to mock if API unavailable)
        llm_summary = self._llm_generate(reasons, is_fraud)
        
        return {
            'text_explanation': explanation,
            'llm_summary': llm_summary,
            'top_features': top_features.to_dict(orient='records')
        }

    def _llm_generate(self, reasons, is_fraud=True):
        """
        Generate natural language explanation using OpenAI API.
        Falls back to mock response if API is not available.
        """
        # Create a detailed prompt for the LLM - полностью на русском языке
        if is_fraud:
            prompt = f"""Проанализируйте следующие факторы, которые привели к пометке этой транзакции как потенциально мошеннической:

Факторы риска:
{chr(10).join(reasons)}

ВАЖНО: Ответьте ТОЛЬКО на русском языке. Все что связано с деньгами указано в валюте тенге. Объясните простыми словами для обычного пользователя, почему эта транзакция была помечена как подозрительная. Будьте конкретны и понятны."""
        else:
            prompt = f"""Проанализируйте следующие факторы, которые подтверждают легитимность этой транзакции:

Факторы:
{chr(10).join(reasons)}

ВАЖНО: Ответьте ТОЛЬКО на русском языке. Все что связано с деньгами указано в валюте тенге. Объясните простыми словами для обычного пользователя, почему эта транзакция считается безопасной. Будьте конкретны и понятны."""

        # Try to use OpenAI API if available
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "Вы - эксперт по обнаружению мошенничества в финансовых транзакциях. ВСЕГДА отвечайте ТОЛЬКО на русском языке. Объясняйте решения модели простым и понятным языком. Никогда не используйте английский язык в ответах. Убедитесь, что ваш ответ является полным и законченным."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                print("Falling back to mock response...")
        
        # Fallback mock response if API is not available
        # Fallback mock response if API is not available
        if is_fraud:
            return "Транзакция помечена как высокорисковая в первую очередь из-за необычной частоты переводов этому получателю и высокой суммы транзакции относительно истории пользователя."
        else:
            return "Транзакция выглядит безопасной. Сумма и получатель соответствуют вашим обычным паттернам активности."

if __name__ == "__main__":
    # Test run
    print("Testing FraudExplainer...")
    try:
        explainer = FraudExplainer()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
