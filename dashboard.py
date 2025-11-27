import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Система обнаружения мошенничества", layout="wide")

API_URL = "http://localhost:8000/predict"

st.title("Система мониторинга мошенничества")

# Sidebar for controls
st.sidebar.header("Настройки")
# Optimal threshold found during training was 0.50
threshold = st.sidebar.slider("Порог блокировки (Threshold)", 0.0, 1.0, 0.5, 0.01)

# LLM Toggle
enable_llm = st.sidebar.checkbox("Включить LLM объяснения (OpenAI)", value=True, help="Если выключено, будут показаны только стандартные шаблоны объяснений.")

# Input method
input_method = st.sidebar.radio("Метод ввода данных", ["Ручной ввод", "Загрузить пример из датасета"])

# Load dataset for context and defaults
@st.cache_data
def load_data():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "featured_dataset.csv"))
    return df

try:
    df = load_data()
    # Get a template row (first legitimate transaction) for default values
    template_row = df[df['target'] == 0].iloc[0].to_dict()
    
    # Get unique values for categorical inputs
    phone_models = df['last_phone_model'].unique().tolist()
    
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    df = None
    template_row = {}
    phone_models = []

input_data = {}

if input_method == "Загрузить пример из датасета":
    if df is not None:
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

else:
    # Manual input
    st.subheader("📝 Ручной ввод параметров транзакции")
    st.info("Значения по умолчанию взяты из типичной легитимной транзакции. Измените ключевые параметры для проверки реакции модели.")
    
    # Start with template
    exclude = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target']
    input_data = {k: v for k, v in template_row.items() if k not in exclude}
    
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Детали транзакции")
            input_data['amount'] = st.number_input("Сумма транзакции", value=float(input_data.get('amount', 0)), help="Сумма перевода в тг.")
            # direction_frequency is actually a COUNT (popularity)
            input_data['direction_frequency'] = st.number_input("Популярность получателя (Direction Count)", value=float(input_data.get('direction_frequency', 0)), help="Сколько раз этот получатель встречался в истории транзакций (популярность).")
            input_data['unique_receivers_from_sender'] = st.number_input("Уникальных получателей (за все время)", value=int(input_data.get('unique_receivers_from_sender', 1)), help="Скольким разным людям клиент отправлял деньги ранее.")
            
            st.markdown("### 📱 Устройство")
            input_data['last_phone_model'] = st.selectbox("Модель телефона", options=phone_models, index=phone_models.index(input_data.get('last_phone_model')) if input_data.get('last_phone_model') in phone_models else 0, help="Модель телефона, с которого совершается перевод.")
            
            # os_frequency is actually a COUNT (popularity)
            input_data['os_frequency'] = st.number_input("Популярность ОС (OS Count)", value=float(input_data.get('os_frequency', 0)), help="Сколько клиентов используют эту версию ОС (чем меньше, тем подозрительнее).")
            
            input_data['monthly_os_changes'] = st.number_input("Смен ОС за месяц", value=int(input_data.get('monthly_os_changes', 0)), help="Сколько раз менялась версия ОС на устройстве за последний месяц.")

        with col2:
            st.markdown("### 📊 История и Поведение")
            input_data['amount_mean_30d'] = st.number_input("Средняя сумма (30 дней)", value=float(input_data.get('amount_mean_30d', 0)), help="Средняя сумма переводов клиента за последние 30 дней.")
            input_data['amount_mean_7d'] = st.number_input("Средняя сумма (7 дней)", value=float(input_data.get('amount_mean_7d', 0)), help="Средняя сумма переводов клиента за последние 7 дней.")
            input_data['avg_login_interval_30d'] = st.number_input("Ср. интервал входа (30 дней)", value=float(input_data.get('avg_login_interval_30d', 0)), help="Среднее время между входами в приложение (в секундах).")
            input_data['logins_7d_over_30d_ratio'] = st.number_input("Отношение входов 7д/30д", value=float(input_data.get('logins_7d_over_30d_ratio', 0)), help="Активность за неделю по сравнению с месяцем (резкий рост может указывать на взлом).")
            
            st.markdown("### 🕒 Время")
            # User friendly Day of Week
            days_map = {
                "Понедельник": 0, "Вторник": 1, "Среда": 2, "Четверг": 3, 
                "Пятница": 4, "Суббота": 5, "Воскресенье": 6
            }
            # Try to reverse engineer current cos value to day name (approximate)
            current_cos = float(input_data.get('day_of_week_cos', 0))
            # cos(2*pi*d/7) = val => d = acos(val) * 7 / (2*pi)
            import math
            try:
                # Clip to valid domain for acos
                val = max(-1.0, min(1.0, current_cos))
                day_idx = int(round(math.acos(val) * 7 / (2 * math.pi))) % 7
            except:
                day_idx = 0
                
            day_name = st.selectbox("День недели", options=list(days_map.keys()), index=day_idx, help="День недели транзакции (преобразуется в Cosine признак для модели).")
            
            # Calculate Cosine feature
            import numpy as np
            input_data['day_of_week_cos'] = np.cos(2 * np.pi * days_map[day_name] / 7)
            
        submitted = st.form_submit_button("Анализировать")

if (input_method == "Загрузить пример из датасета" and st.button("Анализировать транзакцию")) or (input_method == "Ручной ввод" and submitted):
    if not input_data:
        st.error("Нет данных для анализа.")
    else:
        try:
            # Handle potential NaN in input_data (JSON doesn't like NaN)
            clean_data = {k: (v if pd.notna(v) else 0) for k, v in input_data.items()}
            
            # Add LLM flag
            clean_data['enable_llm'] = enable_llm
            
            response = requests.post(API_URL, json=clean_data)
            
            if response.status_code == 200:
                result = response.json()
                prob = result['fraud_probability']
                decision = "БЛОКИРОВАТЬ" if prob > threshold else "ПРОПУСТИТЬ"
                
                # Dashboard Layout
                st.divider()
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
