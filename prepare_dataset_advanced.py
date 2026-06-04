#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def prepare_advanced_dataset():
    print("="*70)
    print("   ПОДГОТОВКА ДАТАСЕТА С РАЗДЕЛЕНИЕМ ПО СФЕРАМ")
    print("="*70)
    
    os.makedirs("dataset/sector_data", exist_ok=True)
    
    # Проверяем наличие raw_dataset.csv
    if os.path.exists("dataset/raw_dataset.csv"):
        df = pd.read_csv("dataset/raw_dataset.csv")
        print(f"\n📊 Загружено: {len(df)} строк")
    else:
        print("\n⚠ Файл dataset/raw_dataset.csv не найден")
        print("   Будут созданы синтетические данные")
        df = pd.DataFrame()
    
    crop_data = []
    forest_data = []
    aqua_data = []
    
    # Растениеводство
    if 'Культура' in df.columns:
        for idx, row in df.iterrows():
            crop_name = row.get('Культура')
            if pd.notna(crop_name) and str(crop_name).strip():
                entry = {
                    'latitude': np.random.uniform(35, 60),
                    'longitude': np.random.uniform(30, 140),
                    'productivity_change': row.get('Урожайность', np.random.uniform(70, 150))
                }
                entry.update({
                    'shade_tolerance': np.random.uniform(0.3, 0.7),
                    'optimal_temp': np.random.uniform(15, 25),
                    'water_requirement': np.random.uniform(300, 800),
                    'growing_days': np.random.randint(60, 180)
                })
                crop_data.append(entry)
    
    # Лесное хозяйство
    if 'Woody_Species' in df.columns:
        for idx, row in df.iterrows():
            woody = row.get('Woody_Species')
            if pd.notna(woody) and str(woody).strip():
                entry = {
                    'latitude': np.random.uniform(35, 60),
                    'longitude': np.random.uniform(30, 140),
                    'productivity_change': row.get('Woody_Relative_Yield_vs_Monoculture_%', np.random.uniform(70, 150))
                }
                entry.update({
                    'tree_height': np.random.uniform(5, 30),
                    'canopy_density': np.random.uniform(0.4, 0.9),
                    'growth_rate': np.random.uniform(0.5, 2.0),
                    'wood_density': np.random.uniform(0.4, 0.8)
                })
                forest_data.append(entry)
    
    # Аквакультура
    if 'Aquatic organism' in df.columns:
        for idx, row in df.iterrows():
            aquatic = row.get('Aquatic organism')
            if pd.notna(aquatic) and str(aquatic).strip():
                entry = {
                    'latitude': np.random.uniform(35, 60),
                    'longitude': np.random.uniform(30, 140),
                    'productivity_change': row.get('Aquaculture production (t/ha)', np.random.uniform(70, 150))
                }
                entry.update({
                    'water_temp': np.random.uniform(15, 25),
                    'oxygen_level': np.random.uniform(5, 9),
                    'stocking_density': np.random.uniform(5, 20),
                    'pond_depth': np.random.uniform(1, 5)
                })
                aqua_data.append(entry)
    
    print(f"\n🌾 Растениеводство: {len(crop_data)} записей")
    print(f"🌲 Лесное хозяйство: {len(forest_data)} записей")
    print(f"🐟 Аквакультура: {len(aqua_data)} записей")
    
    # Сохраняем
    if len(crop_data) > 0:
        pd.DataFrame(crop_data).to_csv("dataset/sector_data/crop_farming.csv", index=False)
        print(f"   ✅ Сохранен: dataset/sector_data/crop_farming.csv")
    else:
        print(f"   ⚠ Нет данных для растениеводства, создаю синтетические")
        synthetic_crop = []
        for i in range(50):
            synthetic_crop.append({
                'latitude': np.random.uniform(35, 60),
                'longitude': np.random.uniform(30, 140),
                'productivity_change': np.random.uniform(70, 150),
                'shade_tolerance': np.random.uniform(0.3, 0.7),
                'optimal_temp': np.random.uniform(15, 25),
                'water_requirement': np.random.uniform(300, 800),
                'growing_days': np.random.randint(60, 180)
            })
        pd.DataFrame(synthetic_crop).to_csv("dataset/sector_data/crop_farming.csv", index=False)
        print(f"   ✅ Создано 50 синтетических записей для растениеводства")
    
    if len(forest_data) > 0:
        pd.DataFrame(forest_data).to_csv("dataset/sector_data/forestry.csv", index=False)
        print(f"   ✅ Сохранен: dataset/sector_data/forestry.csv")
    else:
        print(f"   ⚠ Нет данных для лесного хозяйства, создаю синтетические")
        synthetic_forest = []
        for i in range(50):
            synthetic_forest.append({
                'latitude': np.random.uniform(35, 60),
                'longitude': np.random.uniform(30, 140),
                'productivity_change': np.random.uniform(70, 150),
                'tree_height': np.random.uniform(5, 30),
                'canopy_density': np.random.uniform(0.4, 0.9),
                'growth_rate': np.random.uniform(0.5, 2.0),
                'wood_density': np.random.uniform(0.4, 0.8)
            })
        pd.DataFrame(synthetic_forest).to_csv("dataset/sector_data/forestry.csv", index=False)
        print(f"   ✅ Создано 50 синтетических записей для лесного хозяйства")
    
    if len(aqua_data) > 0:
        pd.DataFrame(aqua_data).to_csv("dataset/sector_data/aquaculture.csv", index=False)
        print(f"   ✅ Сохранен: dataset/sector_data/aquaculture.csv")
    else:
        print(f"   ⚠ Нет данных для аквакультуры, создаю синтетические")
        synthetic_aqua = []
        for i in range(50):
            synthetic_aqua.append({
                'latitude': np.random.uniform(35, 60),
                'longitude': np.random.uniform(30, 140),
                'productivity_change': np.random.uniform(70, 150),
                'water_temp': np.random.uniform(15, 25),
                'oxygen_level': np.random.uniform(5, 9),
                'stocking_density': np.random.uniform(5, 20),
                'pond_depth': np.random.uniform(1, 5)
            })
        pd.DataFrame(synthetic_aqua).to_csv("dataset/sector_data/aquaculture.csv", index=False)
        print(f"   ✅ Создано 50 синтетических записей для аквакультуры")
    
    print("\n" + "="*70)
    print("✅ ДАТАСЕТЫ ГОТОВЫ!")
    print("="*70)

if __name__ == "__main__":
    prepare_advanced_dataset()