"""
Автоматический планировщик для периодического переобучения модели
Использование:
1. Установка: py -m pip install schedule
2. Запуск: py scheduler.py

Настройка частоты переобучения:
- schedule.every().day.at("03:00").do(job) - каждый день в 3:00
- schedule.every().week.do(job) - каждую неделю
- schedule.every(3).days.do(job) - каждые 3 дня
"""

import schedule
import time
from datetime import datetime
from retrain_model import ModelRetrainer

def retrain_job():
    """Job to run retraining"""
    print(f"\n{'='*80}")
    print(f"SCHEDULED RETRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        retrainer = ModelRetrainer()
        retrainer.retrain_and_deploy(auto_deploy=True)
        print(f"\n✅ Scheduled retraining completed successfully!")
    except Exception as e:
        print(f"\n❌ Retraining failed: {e}")

# Schedule configuration
# Пример: переобучение каждый день в 3:00 ночи
schedule.every().day.at("03:00").do(retrain_job)

# Для тестирования: каждые 5 минут (раскомментируйте)
# schedule.every(5).minutes.do(retrain_job)

print("[SCHEDULER] Started!")
print("Scheduled jobs:")
for job in schedule.get_jobs():
    print(f"  - {job}")
print("\nPress Ctrl+C to stop.")

# Run scheduler loop
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
