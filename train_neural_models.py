#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ ДЛЯ ТРЕХ СФЕР СЕЛЬСКОГО ХОЗЯЙСТВА
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def get_training_report(model, X_train, y_train, X_test, y_test, features, sector_name):
    """Формирует отчёт о качестве обучения модели"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    report = {
        'sector': sector_name,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'MLPRegressor (нейросеть)',
        'architecture': {
            'hidden_layers': list(model.hidden_layer_sizes),
            'activation': model.activation,
            'solver': model.solver,
            'max_iter': model.max_iter
        },
        'training_data': {
            'n_samples_train': len(y_train),
            'n_samples_test': len(y_test),
            'total_samples': len(y_train) + len(y_test),
            'features': features
        },
        'metrics': {
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred))
        }
    }
    
    return report

def train_neural_models(force_retrain=False):
    """Обучение нейросетевых моделей для каждой сферы"""
    
    print("="*70)
    print("   ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ (MLPRegressor)")
    print("="*70)
    
    os.makedirs("models", exist_ok=True)
    
    sector_configs = {
        'crop_farming': {
            'name': 'Растениеводство',
            'features': ['latitude', 'longitude', 'shade_tolerance', 'optimal_temp', 'water_requirement', 'growing_days'],
            'icon': '🌾',
            'target': 'productivity_change',
            'hidden_layers': (100, 50, 25),
            'max_iter': 1000
        },
        'aquaculture': {
            'name': 'Аквакультура',
            'features': ['latitude', 'longitude', 'water_temp', 'oxygen_level', 'stocking_density', 'pond_depth'],
            'icon': '🐟',
            'target': 'productivity_change',
            'hidden_layers': (80, 40, 20),
            'max_iter': 1000
        },
        'forestry': {
            'name': 'Лесное хозяйство',
            'features': ['latitude', 'longitude', 'tree_height', 'canopy_density', 'growth_rate', 'wood_density'],
            'icon': '🌲',
            'target': 'productivity_change',
            'hidden_layers': (80, 40, 20),
            'max_iter': 1000
        }
    }
    
    models = {}
    reports = {}
    
    for sector, config in sector_configs.items():
        print(f"\n{config['icon']} {config['name']} ({sector})")
        print("-" * 50)
        
        sector_path = f"dataset/sector_data/{sector}.csv"
        
        if not os.path.exists(sector_path):
            print(f"   ⚠ Файл {sector_path} не найден, пропускаем")
            continue
        
        sector_df = pd.read_csv(sector_path)
        sector_df = sector_df.dropna()
        
        if config['target'] in sector_df.columns:
            sector_df[config['target']] = sector_df[config['target']].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
        
        print(f"   📊 Загружено записей: {len(sector_df)}")
        
        if len(sector_df) < 20:
            print(f"   ⚠ Недостаточно данных для обучения нейросети (нужно минимум 20)")
            continue
        
        available_features = [f for f in config['features'] if f in sector_df.columns]
        X = sector_df[available_features].values
        y = sector_df[config['target']].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = MLPRegressor(
            hidden_layer_sizes=config['hidden_layers'],
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=config['max_iter'],
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
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
        print(f"      • Итераций:  {model.n_iter_}")
        
        model_path = f"models/neural_model_{sector}.pkl"
        scaler_path = f"models/neural_scaler_{sector}.pkl"
        report_path = f"models/neural_report_{sector}.json"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        report = get_training_report(model, X_train, y_train, X_test, y_test, available_features, config['name'])
        report['n_iterations'] = model.n_iter_
        report['loss_curve'] = model.loss_curve_[-10:] if len(model.loss_curve_) > 10 else model.loss_curve_
        report['data_source'] = sector_path
        report['data_rows'] = len(sector_df)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        models[sector] = model
        reports[sector] = report
        
        print(f"   ✅ Модель сохранена: {model_path}")
        print(f"   ✅ Отчёт сохранён: {report_path}")
    
    history_path = "models/training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'last_training': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models_trained': list(reports.keys()),
            'reports': reports
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ ОТЧЕТ ПО ОБУЧЕНИЮ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ")
    print("="*70)
    
    for sector, config in sector_configs.items():
        if sector in reports:
            r = reports[sector]
            print(f"\n{config['icon']} {config['name']}:")
            print(f"   • Обучающих примеров: {r['training_data']['total_samples']}")
            print(f"   • Test MAE: {r['metrics']['test_mae']:.2f}%")
            print(f"   • Test R²:  {r['metrics']['test_r2']:.3f}")
            print(f"   • Итераций: {r['n_iterations']}")
    
    print("\n✅ ВСЕ НЕЙРОСЕТЕВЫЕ МОДЕЛИ УСПЕШНО ОБУЧЕНЫ!")
    print("="*70)
    
    return models, reports

if __name__ == "__main__":
    train_neural_models()