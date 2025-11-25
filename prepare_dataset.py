import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Пути к файлам
BEHAVIORAL_FILE = r"c:\Users\TTR4K\Desktop\hackathon\поведенческие паттерны клиентов.csv"
TRANSACTIONS_FILE = r"c:\Users\TTR4K\Desktop\hackathon\транзакции_в_Мобильном_интернет_Банкинге (1).csv"
OUTPUT_FILE = r"c:\Users\TTR4K\Desktop\hackathon\prepared_dataset.csv"

print("=" * 80)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

# Загрузка данных о поведении клиентов
print("\n1. Загружаем поведенческие паттерны клиентов...")
behavioral_df = pd.read_csv(BEHAVIORAL_FILE, sep=';', encoding='cp1251')
print(f"   Загружено строк: {len(behavioral_df)}")
print(f"   Столбцов: {len(behavioral_df.columns)}")

# Загрузка данных о транзакциях
print("\n2. Загружаем транзакции...")
transactions_df = pd.read_csv(TRANSACTIONS_FILE, sep=';', encoding='cp1251')
print(f"   Загружено строк: {len(transactions_df)}")
print(f"   Столбцов: {len(transactions_df.columns)}")

print("\n" + "=" * 80)
print("ОЧИСТКА И ПЕРЕИМЕНОВАНИЕ СТОЛБЦОВ")
print("=" * 80)

# Создаем короткие имена для столбцов
behavioral_columns_mapping = {
    'Дата совершенной транзакции': 'transdate',
    'Уникальный идентификатор клиента': 'cst_dim_id',
    'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент': 'monthly_os_changes',
    'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент "менял устройство" по логам': 'monthly_phone_model_changes',
    'Модель телефона из самой последней сессии (по времени) перед transdate': 'last_phone_model',
    'Версия ОС из самой последней сессии перед transdate': 'last_os',
    'Количество уникальных логин-сессий (минутных тайм-слотов) за последние 7 дней до transdate': 'logins_last_7_days',
    'Количество уникальных логин-сессий за последние 30 дней до transdate': 'logins_last_30_days',
    'Среднее число логинов в день за последние 7 дней: logins_last_7_days / 7': 'login_frequency_7d',
    'Среднее число логинов в день за последние 30 дней: logins_last_30_days / 30': 'login_frequency_30d',
    'Относительное изменение частоты логинов за 7 дней к средней частоте за 30 дней:\n(freq7d−freq30d)/freq30d(freq_{7d} - freq_{30d}) / freq_{30d}(freq7d−freq30d)/freq30d — показывает, стал клиент заходить чаще или реже недавно': 'freq_change_7d_vs_mean',
    'Доля логинов за 7 дней от логинов за 30 дней': 'logins_7d_over_30d_ratio',
    'Средний интервал (в секундах) между соседними сессиями за последние 30 дней': 'avg_login_interval_30d',
    'Стандартное отклонение интервалов между логинами за 30 дней (в секундах), измеряет разброс интервалов': 'std_login_interval_30d',
    'Дисперсия интервалов между логинами за 30 дней (в секундах?), ещё одна мера разброса': 'var_login_interval_30d',
    'Экспоненциально взвешенное среднее интервалов между логинами за 7 дней, где более свежие сессии имеют больший вес (коэффициент затухания 0.3)': 'ewm_login_interval_7d',
    'Показатель "взрывности" логинов: (std−mean)/(std+mean)(std - mean)/(std + mean)(std−mean)/(std+mean) для интервалов': 'burstiness_login_interval',
    'Fano-factor интервалов: variance / mean': 'fano_factor_login_interval',
    'Z-скор среднего интервала за последние 7 дней относительно среднего за 30 дней: насколько сильно недавние интервалы отличаются от типичных, в единицах стандартного отклонения': 'zscore_avg_login_interval_7d'
}

transactions_columns_mapping = {
    'Уникальный идентификатор клиента': 'cst_dim_id',
    'Дата совершенной транзакции': 'transdate',
    'Дата и время совершенной транзакции': 'transdatetime',
    'Сумма совершенного перевода': 'amount',
    'Уникальный идентификатор транзакции': 'docno',
    'Зашифрованный идентификатор получателя/destination транзакции': 'direction',
    'Размеченные транзакции(переводы), где 1 - мошенническая операция , 0 - чистая': 'target'
}

# Переименовываем столбцы
print("\n3. Переименовываем столбцы...")
behavioral_df.rename(columns=behavioral_columns_mapping, inplace=True)
transactions_df.rename(columns=transactions_columns_mapping, inplace=True)

# Удаляем первую строку (заголовки на английском)
print("\n4. Удаляем строки с дублирующимися заголовками...")
behavioral_df = behavioral_df[behavioral_df['cst_dim_id'] != 'cst_dim_id']
transactions_df = transactions_df[transactions_df['cst_dim_id'] != 'cst_dim_id']

print(f"   Поведенческие данные: {len(behavioral_df)} строк")
print(f"   Транзакции: {len(transactions_df)} строк")

print("\n" + "=" * 80)
print("ОБРАБОТКА ДАННЫХ")
print("=" * 80)

# Очистка строковых значений от кавычек
print("\n5. Очищаем строковые значения от кавычек...")
for col in transactions_df.columns:
    if transactions_df[col].dtype == 'object':
        transactions_df[col] = transactions_df[col].str.strip("'")

for col in behavioral_df.columns:
    if behavioral_df[col].dtype == 'object':
        behavioral_df[col] = behavioral_df[col].str.strip("'")

# Преобразование типов данных
print("\n6. Преобразуем типы данных...")

# Для транзакций
transactions_df['cst_dim_id'] = transactions_df['cst_dim_id'].astype(str)
transactions_df['amount'] = pd.to_numeric(transactions_df['amount'], errors='coerce')
transactions_df['target'] = pd.to_numeric(transactions_df['target'], errors='coerce').astype(int)
transactions_df['docno'] = transactions_df['docno'].astype(str)
transactions_df['direction'] = transactions_df['direction'].astype(str)

# Преобразование дат
transactions_df['transdate'] = pd.to_datetime(transactions_df['transdate'], errors='coerce')
transactions_df['transdatetime'] = pd.to_datetime(transactions_df['transdatetime'], errors='coerce')

# Для поведенческих данных
behavioral_df['cst_dim_id'] = behavioral_df['cst_dim_id'].astype(str)
behavioral_df['transdate'] = pd.to_datetime(behavioral_df['transdate'], errors='coerce')

# Преобразуем числовые столбцы
numeric_cols = [col for col in behavioral_df.columns if col not in ['cst_dim_id', 'transdate', 'last_phone_model', 'last_os']]
for col in numeric_cols:
    behavioral_df[col] = pd.to_numeric(behavioral_df[col], errors='coerce')

print(f"   Распределение целевой переменной:")
print(f"   - Чистые транзакции (0): {(transactions_df['target'] == 0).sum()}")
print(f"   - Мошеннические (1): {(transactions_df['target'] == 1).sum()}")

print("\n" + "=" * 80)
print("ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
print("=" * 80)

# Объединяем датасеты
print("\n7. Объединяем датасеты по ключам (cst_dim_id, transdate)...")
merged_df = pd.merge(
    transactions_df,
    behavioral_df,
    on=['cst_dim_id', 'transdate'],
    how='left'
)

print(f"   Результат объединения: {len(merged_df)} строк, {len(merged_df.columns)} столбцов")

print("\n" + "=" * 80)
print("СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ")
print("=" * 80)

print("\n8. Создаем временные признаки...")
# Временные признаки из transdatetime
merged_df['hour'] = merged_df['transdatetime'].dt.hour
merged_df['day_of_week'] = merged_df['transdatetime'].dt.dayofweek
merged_df['day_of_month'] = merged_df['transdatetime'].dt.day
merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
merged_df['is_night'] = ((merged_df['hour'] >= 22) | (merged_df['hour'] <= 6)).astype(int)

print("\n9. Создаем признаки на основе суммы транзакции...")
# Признаки на основе суммы
merged_df['amount_log'] = np.log1p(merged_df['amount'])
merged_df['is_round_amount'] = (merged_df['amount'] % 1000 == 0).astype(int)

print("\n10. Кодируем категориальные признаки...")
# Frequency encoding для получателя
direction_counts = merged_df['direction'].value_counts()
merged_df['direction_frequency'] = merged_df['direction'].map(direction_counts)

# Frequency encoding для модели телефона и ОС
phone_counts = merged_df['last_phone_model'].value_counts()
merged_df['phone_model_frequency'] = merged_df['last_phone_model'].map(phone_counts)

os_counts = merged_df['last_os'].value_counts()
merged_df['os_frequency'] = merged_df['last_os'].map(os_counts)

print("\n" + "=" * 80)
print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
print("=" * 80)

print("\n11. Проверяем пропущенные значения...")
missing_values = merged_df.isnull().sum()
missing_percent = (missing_values / len(merged_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Столбец': missing_values.index,
    'Пропусков': missing_values.values,
    'Процент': missing_percent.values
})
missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('Пропусков', ascending=False)

if len(missing_df) > 0:
    print(f"\n   Найдено столбцов с пропусками: {len(missing_df)}")
    print(missing_df.to_string(index=False))
else:
    print("   ✓ Пропущенных значений не найдено!")

print("\n" + "=" * 80)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТА")
print("=" * 80)

print(f"\n12. Сохраняем подготовленный датасет в {OUTPUT_FILE}...")
merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n✓ ГОТОВО!")
print(f"\nИтоговый датасет:")
print(f"  - Строк: {len(merged_df)}")
print(f"  - Столбцов: {len(merged_df.columns)}")
print(f"  - Размер файла: {round(merged_df.memory_usage(deep=True).sum() / 1024 / 1024, 2)} MB")

print(f"\nСписок всех признаков ({len(merged_df.columns)}):")
for i, col in enumerate(merged_df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("СТАТИСТИКА ПО ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("=" * 80)
print(f"\nРаспределение:")
print(merged_df['target'].value_counts().to_string())
print(f"\nДоля мошеннических транзакций: {(merged_df['target'].sum() / len(merged_df) * 100):.2f}%")
