import pandas as pd
import os

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

files = [
    os.path.join(SCRIPT_DIR, "поведенческие паттерны клиентов.csv"),
    os.path.join(SCRIPT_DIR, "транзакции_в_Мобильном_интернет_Банкинге (1).csv")
]

for f in files:
    print(f"\n--- Analyzing {os.path.basename(f)} ---")
    try:
        # Peek at the first line to guess delimiter
        # We'll try reading with python's open first to check the first line structure
        try:
            with open(f, 'r', encoding='utf-8') as file:
                first_line = file.readline()
            encoding = 'utf-8'
        except UnicodeDecodeError:
            with open(f, 'r', encoding='cp1251') as file:
                first_line = file.readline()
            encoding = 'cp1251'
            
        if ';' in first_line:
            sep = ';'
        else:
            sep = ',' 
        
        print(f"Detected delimiter: '{sep}', Encoding: {encoding}")
        
        df = pd.read_csv(f, sep=sep, encoding=encoding)
        
    except Exception as e:
        print(f"Error reading file: {e}")
        continue

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 2 rows:")
    print(df.head(2).to_string())
