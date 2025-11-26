import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Система обнаружения мошенничества", layout="wide")

API_URL = "http://localhost:8000/predict"

st.title("🛡️ Система мониторинга мошенничества")

# Sidebar for controls
st.sidebar.header("Настройки")
# Optimal threshold found during training was 0.50
threshold = st.sidebar.slider("Порог блокировки (Threshold)", 0.0, 1.0, 0.5, 0.01)

# Input method
input_method = st.sidebar.radio("Метод ввода данных", ["Ручной ввод", "Загрузить пример из датасета"])

input_data = {}

if input_method == "Загрузить пример из датасета":
    # Load a sample from the featured dataset
    try:
        # Получаем директорию текущего скрипта
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "featured_dataset.csv"))
        # Pick a random row or let user select by ID
        sample_id = st.sidebar.number_input("Выберите индекс транзакции", 0, len(df)-1, 0)
        row = df.iloc[sample_id]
        
        # Filter columns to match what the API expects (exclude target, IDs, dates)
        exclude = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target']
        input_data = row.drop(labels=exclude).to_dict()
        
        st.info(f"Загружена транзакция клиента {row['cst_dim_id']} от {row['transdate']}")
        
        # Show ground truth
        is_fraud_truth = row['target'] == 1
        if is_fraud_truth:
            st.error("⚠️ Истинное значение: МОШЕННИЧЕСТВО (FRAUD)")
        else:
            st.success("✅ Истинное значение: ЛЕГИТИМНАЯ (LEGITIMATE)")
            
    except Exception as e:
        st.error(f"Ошибка загрузки датасета: {e}")

else:
    # Manual input (simplified for demo)
    st.warning("Ручной ввод требует множества признаков. Для демо используются значения по умолчанию.")
    input_data = {
        "amount": 50000,
        "time_since_last_trans": 60, # 1 minute
        "amount_mean_30d": 5000,
        "amount_zscore_30d": 10.0,
        "device_changed": 1,
        "unique_senders_to_receiver": 50,
        # Add other necessary fields with defaults...
    }

if st.button("Анализировать транзакцию"):
    if not input_data:
        st.error("Нет данных для анализа.")
    else:
        try:
            # Handle potential NaN in input_data (JSON doesn't like NaN)
            clean_data = {k: (v if pd.notna(v) else 0) for k, v in input_data.items()}
            
            response = requests.post(API_URL, json=clean_data)
            
            if response.status_code == 200:
                result = response.json()
                prob = result['fraud_probability']
                decision = "БЛОКИРОВАТЬ" if prob > threshold else "ПРОПУСТИТЬ"
                
                # Dashboard Layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Оценка риска")
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Вероятность мошенничества (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prob > threshold else "green"},
                            'steps': [
                                {'range': [0, threshold*100], 'color': "lightgreen"},
                                {'range': [threshold*100, 100], 'color': "salmon"}],
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    if decision == "БЛОКИРОВАТЬ":
                        st.error(f"⛔ РЕШЕНИЕ: {decision}")
                    else:
                        st.success(f"✅ РЕШЕНИЕ: {decision}")
                
                with col2:
                    st.subheader("Анализ и объяснение")
                    
                    expl = result['explanation']
                    st.markdown(f"**Резюме модели:**\n> {expl['llm_summary']}")
                    
                    st.markdown("**Ключевые факторы риска:**")
                    st.text(expl['text_explanation'])
                    
                    # Feature Importance Plot (Top 3)
                    top_feats = pd.DataFrame(expl['top_features'])
                    st.bar_chart(top_feats.set_index('feature')['shap_value'])
                    
            else:
                st.error(f"Ошибка API: {response.text}")
                
        except Exception as e:
            st.error(f"Ошибка соединения: {e}. Запущен ли API (app.py)?")

st.markdown("---")
st.caption("Система обнаружения мошенничества MVP | Powered by LightGBM & SHAP")
