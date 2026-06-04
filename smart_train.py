#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
УМНОЕ ОБУЧЕНИЕ: автоматический выбор модели
- Если данных > 100 → обучаем нейросеть + Random Forest
- Если данных 20-100 → обучаем только Random Forest
- Если данных < 20 → используем fallback (значения по умолчанию)
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def get_data_size(sector_path):
    """Возвращает количество строк в датасете"""
    if not os.path.exists(sector_path):
        return 0
    df = pd.read_csv(sector_path)
    return len(df)

def train_random_forest(X_train, y_train, X_test, y_test, features, sector_name):
    """Обучение Random Forest"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    return model, test_mae, test_r2

def train_neural_network(X_train, y_train, X_test, y_test, features, sector_name):
    """Обучение нейросети"""
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    return model, test_mae, test_r2

def smart_train():
    """Умное обучение: выбор модели по размеру данных"""
    
    print("="*70)
    print("   УМНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ (автовыбор по размеру данных)")
    print("="*70)
    
    os.makedirs("models", exist_ok=True)
    
    sector_configs = {
        'crop_farming': {
            'name': 'Растениеводство',
            'features': ['latitude', 'longitude', 'shade_tolerance', 'optimal_temp', 'water_requirement', 'growing_days'],
            'icon': '🌾',
            'target': 'productivity_change',
            'min_samples': 3
        },
        'aquaculture': {
            'name': 'Аквакультура',
            'features': ['latitude', 'longitude', 'water_temp', 'oxygen_level', 'stocking_density', 'pond_depth'],
            'icon': '🐟',
            'target': 'productivity_change',
            'min_samples': 3
        },
        'forestry': {
            'name': 'Лесное хозяйство',
            'features': ['latitude', 'longitude', 'tree_height', 'canopy_density', 'growth_rate', 'wood_density'],
            'icon': '🌲',
            'target': 'productivity_change',
            'min_samples': 3
        }
    }
    
    results = {}
    
    for sector, config in sector_configs.items():
        print(f"\n{config['icon']} {config['name']} ({sector})")
        print("-" * 50)
        
        sector_path = f"dataset/sector_data/{sector}.csv"
        data_size = get_data_size(sector_path)
        
        print(f"   📊 Размер данных: {data_size} строк")
        
        # 🔥 ГЛАВНАЯ ЛОГИКА ВЫБОРА МОДЕЛИ
        if data_size < config['min_samples']:
            print(f"   ⚠ Данных недостаточно (< {config['min_samples']})")
            print(f"   → Используется FALLBACK (значения по умолчанию)")
            results[sector] = {
                'model_type': 'fallback',
                'trained': False,
                'message': f'Недостаточно данных (требуется {config["min_samples"]}+)',
                'data_size': data_size
            }
            continue
        
        # Загружаем и подготавливаем данные
        df = pd.read_csv(sector_path)
        df = df.dropna()
        
        if config['target'] in df.columns:
            df[config['target']] = df[config['target']].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
        
        available_features = [f for f in config['features'] if f in df.columns]
        X = df[available_features].values
        y = df[config['target']].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 🔥 ВЫБОР МОДЕЛИ ПО РАЗМЕРУ ДАННЫХ
        if data_size >= 100:
            print(f"   ✅ Данных достаточно ({data_size} >= 100)")
            print(f"   → Обучаем НЕЙРОСЕТЬ + Random Forest")
            
            # Обучаем нейросеть
            neural_model, neural_mae, neural_r2 = train_neural_network(
                X_train, y_train, X_test, y_test, available_features, config['name']
            )
            
            # Обучаем Random Forest
            rf_model, rf_mae, rf_r2 = train_random_forest(
                X_train, y_train, X_test, y_test, available_features, config['name']
            )
            
            # Выбираем лучшую модель по R²
            if neural_r2 >= rf_r2:
                best_model = neural_model
                best_type = 'neural_network'
                best_mae = neural_mae
                best_r2 = neural_r2
                print(f"   🏆 Лучшая: Нейросеть (R²={neural_r2:.3f} > {rf_r2:.3f})")
            else:
                best_model = rf_model
                best_type = 'random_forest'
                best_mae = rf_mae
                best_r2 = rf_r2
                print(f"   🏆 Лучшая: Random Forest (R²={rf_r2:.3f} > {neural_r2:.3f})")
            
            # Сохраняем обе модели
            joblib.dump(neural_model, f'models/neural_model_{sector}.pkl')
            joblib.dump(rf_model, f'models/model_{sector}.pkl')
            
        elif data_size >= 50:
            print(f"   ✅ Данных средне ({data_size} >= 50)")
            print(f"   → Обучаем только Random Forest")
            
            rf_model, rf_mae, rf_r2 = train_random_forest(
                X_train, y_train, X_test, y_test, available_features, config['name']
            )
            best_model = rf_model
            best_type = 'random_forest'
            best_mae = rf_mae
            best_r2 = rf_r2
            
            joblib.dump(rf_model, f'models/model_{sector}.pkl')
            
        else:  # 20-49 строк
            print(f"   ✅ Данных достаточно ({data_size} >= 20)")
            print(f"   → Обучаем Random Forest (упрощённый)")
            
            # Упрощённый Random Forest для малых данных
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            best_mae = mean_absolute_error(y_test, y_pred_test)
            best_r2 = r2_score(y_test, y_pred_test)
            best_model = model
            best_type = 'random_forest_simple'
            
            joblib.dump(model, f'models/model_{sector}.pkl')
        
        # Сохраняем scaler
        joblib.dump(scaler, f'models/scaler_{sector}.pkl')
        
        # Сохраняем отчёт
        report = {
            'sector': config['name'],
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_size': data_size,
            'model_used': best_type,
            'features': available_features,
            'metrics': {
                'test_mae': float(best_mae),
                'test_r2': float(best_r2)
            }
        }
        
        with open(f'models/smart_report_{sector}.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        results[sector] = {
            'model_type': best_type,
            'trained': True,
            'data_size': data_size,
            'test_mae': best_mae,
            'test_r2': best_r2
        }
        
        print(f"   ✅ Обучено: {best_type}")
        print(f"   📈 Test MAE: {best_mae:.2f}%, Test R²: {best_r2:.3f}")
    
    # Итоговый отчёт
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ ОТЧЁТ ПО УМНОМУ ОБУЧЕНИЮ")
    print("="*70)
    
    for sector, config in sector_configs.items():
        if sector in results:
            r = results[sector]
            print(f"\n{config['icon']} {config['name']}:")
            print(f"   • Размер данных: {r['data_size']} строк")
            print(f"   • Модель: {r['model_type']}")
            if r['trained']:
                print(f"   • R²: {r['test_r2']:.3f}, MAE: {r['test_mae']:.2f}%")
            else:
                print(f"   • Статус: {r.get('message', 'не обучена')}")
    
    print("\n✅ УМНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    
    return results

if __name__ == "__main__":
    smart_train()