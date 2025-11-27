# Система Обнаружения Мошенничества (Fraud Detection System)

## Обзор Проекта
Данный проект представляет собой MVP (Minimum Viable Product) системы обнаружения мошеннических транзакций в банковской сфере. Система использует алгоритмы машинного обучения (LightGBM) для анализа транзакций в реальном времени и предоставляет объяснения принятых решений с помощью SHAP значений и интеграции с LLM (OpenAI).

## Основные Возможности
- **Анализ транзакций**: Оценка вероятности мошенничества для каждой транзакции.
- **Объяснимость (Explainability)**: Генерация понятных объяснений причин блокировки или пропуска транзакции (SHAP + LLM).
- **API**: REST API на базе FastAPI для интеграции.
- **Дашборд**: Интерактивный веб-интерфейс на Streamlit для мониторинга и ручного анализа.
- **Feature Engineering**: Сложная система генерации признаков, включая временные окна, поведенческие паттерны и графовые признаки.

## Установка и Настройка

### Предварительные требования
- Python 3.10+
- Git

### Установка
1. **Клонируйте репозиторий:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Настройка переменных окружения:**
   Создайте файл `.env` в корне проекта и добавьте ваш API ключ OpenAI (для генерации текстовых объяснений):
   ```env
   OPENAI_API_KEY=sk-...
   ```

## Архитектура и Структура Проекта

### Структура Файлов
- `prepare_dataset.py`: Скрипт первичной обработки и объединения сырых данных (транзакции + профили).
- `feature_engineering.py`: Генерация признаков (rolling windows, velocity, graph features).
- `train_model.py`: Обучение модели LightGBM с подбором гиперпараметров и кросс-валидацией.
- `explainability.py`: Модуль для генерации объяснений (SHAP) и интеграции с LLM.
- `app.py`: FastAPI сервис, предоставляющий эндпоинт `/predict`.
- `dashboard.py`: Streamlit приложение для визуализации и тестирования.

### Пайплайн Данных
1. **Raw Data** (`транзакции...csv`, `поведенческие...csv`) -> `prepare_dataset.py` -> `prepared_dataset.csv`
2. `prepared_dataset.csv` -> `feature_engineering.py` -> `featured_dataset.csv`
3. `featured_dataset.csv` -> `train_model.py` -> `lgbm_model.pkl` (Модель) & `model_metrics.json` (Метрики)

## Использование

### 1. Подготовка Данных и Обучение (если нужно переобучить)
```bash
python prepare_dataset.py
python feature_engineering.py
python train_model.py
```

### 2. Запуск API
Запустите сервер FastAPI:
```bash
uvicorn app:app --reload
```
API будет доступно по адресу: `http://localhost:8000`
Документация Swagger UI: `http://localhost:8000/docs`

**Пример запроса:**
```json
POST /predict
{
  "amount": 5000,
  "time_since_last_trans": 300,
  "amount_mean_30d": 4500,
  ... (другие признаки)
}
```

### 3. Запуск Дашборда
В новом терминале запустите Streamlit:
```bash
streamlit run dashboard.py
```
Дашборд откроется в браузере. Вы можете:
- Выбрать случайную транзакцию из датасета.
- Ввести параметры транзакции вручную.
- Увидеть вероятность мошенничества и текстовое объяснение причин.

## Метрики Модели
Текущая модель (LightGBM) показывает следующие метрики на валидации:
- **ROC-AUC**: ~0.85+
- **Precision/Recall**: Сбалансированы для минимизации пропуска фрода при приемлемом уровне ложных срабатываний.

## API Reference
### `POST /predict`
Принимает JSON с признаками транзакции. Возвращает:
- `fraud_probability`: Вероятность (0-1).
- `is_fraud`: Предсказание (True/False).
- `decision`: Рекомендация (BLOCK/ALLOW).
- `explanation`: Объект с объяснением (текст + SHAP values).
