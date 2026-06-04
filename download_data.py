#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
СКАЧИВАНИЕ ВСЕХ БАЗ ДАННЫХ АГРИВОЛЬТАИКИ
"""

import pandas as pd
import os

def download_all_data():
    """Скачивание всех таблиц"""
    
    print("="*70)
    print("      СКАЧИВАНИЕ БАЗ ДАННЫХ АГРИВОЛЬТАИКИ")
    print("="*70)
    
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/sector_data", exist_ok=True)
    
    urls = [
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSrhKbZjKkGqSytiGNDciKhx5LYm9bubt-WFqbuASR48nvkC8yEg16fZKnbRgXeMdNPSAVJbZlnw-Sc/pub?output=csv",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR0wSk6qgGv0cZxkc4zRKwJso5edQ88ivsUDACEXts43BR4gZRfcGJ3JzjpPjYYpLnUvAe_mJJO0UJg/pub?output=csv",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSdqh15EIG4xIACOBaFtWbKamveCBj7QVPq2vkyVhjk0_F5mrrlRqMJROWCXXkKDr3aBzqA3GCNNHyt/pub?output=csv"
    ]
    
    all_dfs = []
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"\n📥 Загрузка таблицы {i}...")
            df = pd.read_csv(url)
            print(f"   ✓ Загружено строк: {len(df)}")
            print(f"   ✓ Колонки: {list(df.columns)[:5]}...")
            all_dfs.append(df)
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_path = "dataset/raw_dataset.csv"
        combined.to_csv(output_path, index=False)
        
        print("\n" + "="*70)
        print(f"✅ УСПЕХ! Объединенный датасет:")
        print(f"   • Всего строк: {len(combined)}")
        print(f"   • Всего колонок: {len(combined.columns)}")
        print(f"   • Сохранен: {output_path}")
        print("="*70)
        
        return combined
    else:
        print("\n❌ Не удалось загрузить данные")
        return None

if __name__ == "__main__":
    download_all_data()