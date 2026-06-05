#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
АВТОМАТИЧЕСКОЕ ПЕРЕОБУЧЕНИЕ МОДЕЛЕЙ
Запускается при добавлении новых данных в БД
Использует умное обучение (smart_train) с автовыбором модели
"""

import os
import json
import hashlib
from datetime import datetime

def get_data_hash():
    """Вычисляет хеш всех данных для отслеживания изменений"""
    
    data_files = [
        "dataset/sector_data/crop_farming.csv",
        "dataset/sector_data/aquaculture.csv",
        "dataset/sector_data/forestry.csv"
    ]
    
    hasher = hashlib.md5()
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    
    return hasher.hexdigest()

def load_last_hash():
    """Загружает последний сохранённый хеш"""
    
    hash_file = "models/data_hash.txt"
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return f.read().strip()
    return None

def save_current_hash(hash_value):
    """Сохраняет текущий хеш"""
    
    hash_file = "models/data_hash.txt"
    os.makedirs("models", exist_ok=True)
    with open(hash_file, 'w') as f:
        f.write(hash_value)

def check_data_changed():
    """Проверяет, изменились ли данные"""
    
    current_hash = get_data_hash()
    last_hash = load_last_hash()
    
    if current_hash != last_hash:
        return True, current_hash, last_hash
    return False, current_hash, last_hash

def auto_train():
    """Автоматическое обучение моделей при изменении данных"""
    
    print("="*70)
    print("   АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    changed, current_hash, last_hash = check_data_changed()
    
    if not changed:
        print("\n✅ Данные не изменились. Переобучение не требуется.")
        print(f"   Текущий хеш: {current_hash[:16]}...")
        return False
    
    print(f"\n🔄 Обнаружены изменения в данных!")
    print(f"   Старый хеш: {last_hash[:16] if last_hash else 'None'}...")
    print(f"   Новый хеш: {current_hash[:16]}...")
    print(f"   Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🚀 Запуск умного обучения...")
    
    try:
        from smart_train import smart_train
        results = smart_train()
        
        save_current_hash(current_hash)
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_hash': current_hash,
            'previous_hash': last_hash,
            'models_trained': list(results.keys()),
            'status': 'success',
            'details': results
        }
        
        report_path = "models/auto_train_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n✅ Автоматическое обучение завершено успешно!")
        print(f"   Отчёт сохранён: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        
        error_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error': str(e),
            'status': 'failed'
        }
        
        with open("models/auto_train_error.json", 'w', encoding='utf-8') as f:
            json.dump(error_report, f, ensure_ascii=False, indent=2)
        
        return False

def get_training_info():
    """Возвращает информацию о последнем обучении"""
    
    info = {
        'last_training': None,
        'data_hash': None,
        'data_changed_since_last': None,
        'models_status': {}
    }
    
    # Загружаем последний отчёт об обучении
    report_path = "models/auto_train_report.json"
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            info['last_training'] = report.get('timestamp')
            info['models_status'] = report.get('details', {})
    
    # Загружаем хеш данных
    last_hash = load_last_hash()
    info['data_hash'] = last_hash
    
    # Проверяем, изменились ли данные
    current_hash = get_data_hash()
    info['data_changed_since_last'] = (current_hash != last_hash) if last_hash else True
    
    return info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true', help='Показать статус обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное переобучение')
    args = parser.parse_args()
    
    if args.status:
        info = get_training_info()
        print("\n📊 СТАТУС ОБУЧЕНИЯ")
        print("="*50)
        print(f"Последнее обучение: {info['last_training'] or 'никогда'}")
        print(f"Хеш данных: {info['data_hash'][:16] if info['data_hash'] else 'None'}...")
        print(f"Данные изменились: {'Да' if info['data_changed_since_last'] else 'Нет'}")
        print(f"\nМодели:")
        for sector, status in info['models_status'].items():
            model_type = status.get('model_type', '?')
            data_size = status.get('data_size', '?')
            if status.get('trained'):
                r2 = status.get('test_r2', '?')
                mae = status.get('test_mae', '?')
                print(f"  • {sector}: {model_type} | данные={data_size} | R²={r2} | MAE={mae}")
            else:
                print(f"  • {sector}: {model_type} | данные={data_size}")
    elif args.force:
        auto_train()
    else:
        auto_train()
