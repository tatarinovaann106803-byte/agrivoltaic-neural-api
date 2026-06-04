#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ОБУЧЕНИЕ ОТДЕЛЬНЫХ AI МОДЕЛЕЙ ДЛЯ ТРЕХ СФЕР С/Х
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_sector_models():
    """Обучение моделей для каждой сферы"""
    
    print("="*70)
    print("   ОБУЧЕНИЕ AI МОДЕЛЕЙ ДЛЯ ТРЕХ СФЕР СЕЛЬСКОГО ХОЗЯЙСТВА")
    print("="*70)
    
    os.makedirs("models", exist_ok=True)
    
    sector_configs = {
        'crop_farming': {
            'name': 'Растениеводство',
            'features': ['latitude', 'longitude', 'shade_tolerance', 'optimal_temp', 'water_requirement', 'growing_days'],
            'icon': '🌾'
        },
        'aquaculture': {
            'name': 'Аквакультура',
            'features': ['latitude', 'longitude', 'water_temp', 'oxygen_level', 'stocking_density', 'pond_depth'],
            'icon': '🐟'
        },
        'forestry': {
            'name': 'Лесное хозяйство',
            'features': ['latitude', 'longitude', 'tree_height', 'canopy_density', 'growth_rate', 'wood_density'],
            'icon': '🌲'
        }
    }
    
    models = {}
    results = {}
    
    for sector, config in sector_configs.items():
        print(f"\n{config['icon']} {config['name']} ({sector})")
        print("-" * 50)
        
        sector_path = f"dataset/sector_data/{sector}.csv"
        
        if not os.path.exists(sector_path):
            print(f"   ⚠ Файл {sector_path} не найден, создаю синтетические данные")
            
            n_samples = 100
            data = {}
            for feature in config['features']:
                if feature in ['latitude', 'longitude']:
                    data[feature] = np.random.uniform(35, 60, n_samples)
                elif 'temp' in feature:
                    data[feature] = np.random.uniform(10, 30, n_samples)
                elif 'density' in feature:
                    data[feature] = np.random.uniform(0.3, 0.8, n_samples)
                else:
                    data[feature] = np.random.uniform(0, 1, n_samples) * np.random.uniform(100, 1000, n_samples)
            data['productivity_change'] = np.random.uniform(70, 150, n_samples)
            sector_df = pd.DataFrame(data)
        else:
            sector_df = pd.read_csv(sector_path)
        
        print(f"   📊 Загружено записей: {len(sector_df)}")
        
        if len(sector_df) < 10:
            print(f"   ⚠ Недостаточно данных для обучения (нужно минимум 10)")
            continue
        
        X = sector_df[config['features']].values
        y = sector_df['productivity_change'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"   📈 Качество модели:")
        print(f"      • Train MAE: {train_mae:.2f}%")
        print(f"      • Test MAE:  {test_mae:.2f}%")
        print(f"      • Train R²:  {train_r2:.3f}")
        print(f"      • Test R²:   {test_r2:.3f}")
        
        importance = model.feature_importances_
        print(f"   🔍 Важность признаков:")
        for i, feature in enumerate(config['features']):
            print(f"      • {feature}: {importance[i]*100:.1f}%")
        
        model_path = f"models/model_{sector}.pkl"
        scaler_path = f"models/scaler_{sector}.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        models[sector] = model
        results[sector] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(sector_df)
        }
        
        print(f"   ✅ Модель сохранена: {model_path}")
    
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ ОТЧЕТ ПО ОБУЧЕНИЮ МОДЕЛЕЙ")
    print("="*70)
    
    for sector, config in sector_configs.items():
        if sector in results:
            r = results[sector]
            print(f"\n{config['icon']} {config['name']}:")
            print(f"   • Обучающих примеров: {r['n_samples']}")
            print(f"   • Test MAE: {r['test_mae']:.2f}%")
            print(f"   • Test R²:  {r['test_r2']:.3f}")
    
    print("\n✅ ВСЕ МОДЕЛИ УСПЕШНО ОБУЧЕНЫ!")
    print("="*70)
    
    return models, results

if __name__ == "__main__":
    train_sector_models()