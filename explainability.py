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

# Словарь с метаданными признаков: название и описание для LLM
FEATURE_METADATA = {
    'amount': {
        'name': 'Сумма транзакции',
        'description': 'Сумма текущего перевода в тенге. Аномально большие суммы могут быть признаком кражи средств.'
    },
    'time_since_last_trans': {
        'name': 'Время с последней транзакции',
        'description': 'Количество секунд, прошедших с момента предыдущей операции клиента. Очень короткие интервалы (секунды) могут указывать на работу автоматического скрипта (бота).'
    },
    'amount_mean_1d': {
        'name': 'Средняя сумма за 24 часа',
        'description': 'Средняя величина транзакций клиента за последние сутки. Используется для сравнения с текущей суммой.'
    },
    'amount_mean_7d': {
        'name': 'Средняя сумма за 7 дней',
        'description': 'Средняя величина транзакций клиента за последнюю неделю.'
    },
    'amount_mean_30d': {
        'name': 'Средняя сумма за 30 дней',
        'description': 'Средняя величина транзакций клиента за последний месяц. Существенное отклонение текущей суммы от этого значения подозрительно.'
    },
    'amount_std_30d': {
        'name': 'Разброс сумм за 30 дней',
        'description': 'Стандартное отклонение сумм транзакций. Показывает, насколько разнообразны суммы переводов клиента. Низкий разброс означает, что клиент обычно переводит одни и те же суммы.'
    },
    'amount_zscore_30d': {
        'name': 'Аномальность суммы (Z-score)',
        'description': 'Статистический показатель, насколько текущая сумма отличается от обычной для этого клиента. Высокое значение означает сильную аномалию.'
    },
    'trans_count_1h': {
        'name': 'Транзакций за последний час',
        'description': 'Количество операций, совершенных клиентом за последний час. Резкий всплеск активности (много операций подряд) характерен для взлома.'
    },
    'device_changed': {
        'name': 'Смена устройства',
        'description': 'Флаг, указывающий, что транзакция совершается с нового устройства, которое клиент ранее не использовал.'
    },
    'os_changed': {
        'name': 'Смена ОС',
        'description': 'Флаг, указывающий, что операционная система устройства изменилась. Может означать вход с чужого устройства.'
    },
    'unique_senders_to_receiver': {
        'name': 'Уникальных отправителей получателю',
        'description': 'Количество разных людей, которые переводили деньги этому получателю. Высокое значение может указывать на то, что получатель - "дроппер" (обнальщик мошеннических денег).'
    },
    'unique_receivers_from_sender': {
        'name': 'Уникальных получателей у отправителя',
        'description': 'Количество разных людей, которым клиент переводил деньги. Аномально высокое значение может указывать на веерную рассылку средств или взлом.'
    },
    'direction_frequency': {
        'name': 'Популярность получателя',
        'description': 'Как часто этот получатель получает переводы в системе в целом. Перевод на редкого или нового получателя более рискован.'
    },
    'is_night': {
        'name': 'Ночное время',
        'description': 'Транзакция совершена ночью (обычно с 00:00 до 06:00). Ночная активность нетипична для большинства клиентов.'
    },
    'is_weekend': {
        'name': 'Выходной день',
        'description': 'Транзакция совершена в субботу или воскресенье.'
    },
    'is_round_amount': {
        'name': 'Круглая сумма',
        'description': 'Сумма транзакции является круглым числом. Мошенники часто используют круглые суммы для удобства.'
    },
    'amount_log': {
        'name': 'Логарифм суммы',
        'description': 'Математическое преобразование суммы, используемое моделью для нормализации данных.'
    },
    'hour': {
        'name': 'Час суток',
        'description': 'Час, в который была совершена транзакция.'
    },
    'day_of_week': {
        'name': 'День недели',
        'description': 'День недели транзакции.'
    },
    'phone_model_frequency': {
        'name': 'Популярность модели телефона',
        'description': 'Насколько часто встречается данная модель телефона среди всех клиентов.'
    },
    'os_frequency': {
        'name': 'Популярность версии ОС',
        'description': 'Насколько часто встречается данная версия операционной системы.'
    },
    'var_login_interval_30d': {
        'name': 'Вариативность интервалов входа',
        'description': 'Насколько регулярно или хаотично клиент заходит в приложение. Боты часто имеют строгую периодичность (низкая вариативность).'
    },
    'monthly_os_changes': {
        'name': 'Смен ОС за месяц',
        'description': 'Сколько раз менялась операционная система за последний месяц. Частые смены подозрительны.'
    },
    'logins_7d_over_30d_ratio': {
        'name': 'Активность (неделя/месяц)',
        'description': 'Отношение количества входов за последнюю неделю к количеству входов за месяц. Резкое изменение активности может быть признаком взлома.'
    },
    'avg_login_interval_30d': {
        'name': 'Средний интервал входа',
        'description': 'Среднее время между входами в приложение.'
    }
}

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

    def generate_explanation(self, row_data, shap_values, feature_names, is_fraud=True, enable_llm=True):
        # Generate a text explanation based on top SHAP features
        # row_data: pandas Series or dict of feature values
        # shap_values: numpy array of shap values for this row
        
        # Friendly names mapping removed - using FEATURE_METADATA

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
            
            # Get metadata
            meta = FEATURE_METADATA.get(feat, {'name': feat, 'description': 'Фактор, влияющий на оценку риска транзакции.'})
            friendly_name = meta['name']
            description = meta['description']
            
            # Format value for readability
            if isinstance(val, (int, float)):
                if abs(val) > 1000000:
                    # For huge numbers like variance, use scientific notation or simplified text
                    if 'var_login' in feat:
                        # Convert variance to std dev in hours for readability
                        import math
                        try:
                            std_hours = math.sqrt(val) / 3600
                            val_str = f"{val:.2e} (StdDev: ~{std_hours:.1f} часов)"
                        except:
                            val_str = f"{val:.2e}"
                    else:
                        val_str = f"{val:.2e}"
                elif isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
            else:
                val_str = str(val)

            # Determine direction
            if shap_val > 0:
                direction = "ПОВЫШАЕТ риск"
            else:
                direction = "СНИЖАЕТ риск (говорит о безопасности)"
                
            # Construct detailed reason for LLM
            reason_entry = (
                f"Фактор: {friendly_name}\n"
                f"   Значение: {val_str}\n"
                f"   Влияние: {direction} (SHAP: {shap_val:.2f})\n"
                f"   Что это значит: {description}"
            )
            reasons.append(reason_entry)
            
            # Simple text for fallback/log
            explanation += f"- {friendly_name}: {val_str} ({direction})\n"
        
        # Real LLM call (with fallback to mock if API unavailable)
        # ONLY call if enabled AND is_fraud (to save costs/time)
        if enable_llm and is_fraud:
            llm_summary = self._llm_generate(reasons, is_fraud)
        else:
            if not is_fraud:
                llm_summary = "Транзакция выглядит безопасной. Сумма и получатель соответствуют вашим обычным паттернам активности."
            else:
                llm_summary = "LLM объяснение отключено. См. факторы риска выше."
        
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
            prompt = f"""Вы - эксперт по кибербезопасности банка. Проанализируйте транзакцию, которая была помечена системой антифрода как ПОДОЗРИТЕЛЬНАЯ.

Ниже приведены ключевые факторы, повлиявшие на это решение. Для каждого фактора дано описание того, что он означает, и как он повлиял на оценку риска.

ФАКТОРЫ РИСКА:
{chr(10).join(reasons)}

ЗАДАЧА:
Объясните клиенту (или оператору колл-центра) простым и понятным языком, почему эта транзакция выглядит подозрительно.
1. Используйте описания факторов, чтобы объяснить суть проблемы. Не просто перечисляйте цифры.
2. Свяжите факторы между собой в логическую историю. Например, "Смена устройства в сочетании с нетипично крупной суммой выглядит как попытка кражи".
3. Будьте убедительны, но вежливы.
4. Ответ должен быть на русском языке. Суммы указывайте в тенге."""
        else:
            prompt = f"""Вы - эксперт по кибербезопасности банка. Проанализируйте транзакцию, которая была признана БЕЗОПАСНОЙ (легитимной).

Ниже приведены ключевые факторы, подтверждающие это.

ФАКТОРЫ БЕЗОПАСНОСТИ:
{chr(10).join(reasons)}

ЗАДАЧА:
Объясните простым языком, почему мы считаем эту операцию безопасной.
1. Укажите, что поведение клиента соответствует его обычным привычкам (если это следует из факторов).
2. Объясните, почему факторы снижают риск (например, "Клиент использует привычное устройство").
3. Ответ должен быть на русском языке."""

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
