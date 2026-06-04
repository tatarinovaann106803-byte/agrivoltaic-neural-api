#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПРЕДСКАЗАНИЕ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ
"""

import joblib
import numpy as np
import os
import json
from datetime import datetime

class AgrivoltaicNeuralPredictor:
    """Класс для предсказания с использованием нейросетей"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.neural_models = {}
        self.neural_scalers = {}
        self.random_forest_models = {}
        self.random_forest_scalers = {}
        self.model_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Загрузка всех моделей (нейросетевых и Random Forest)"""
        
        sectors = ['crop_farming', 'aquaculture', 'forestry']
        
        for sector in sectors:
            neural_model_path = f"{self.models_dir}/neural_model_{sector}.pkl"
            neural_scaler_path = f"{self.models_dir}/neural_scaler_{sector}.pkl"
            
            if os.path.exists(neural_model_path) and os.path.exists(neural_scaler_path):
                try:
                    self.neural_models[sector] = joblib.load(neural_model_path)
                    self.neural_scalers[sector] = joblib.load(neural_scaler_path)
                    
                    report_path = f"{self.models_dir}/neural_report_{sector}.json"
                    if os.path.exists(report_path):
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report = json.load(f)
                            self.model_info[sector] = {
                                'type': 'neural_network',
                                'metrics': report['metrics'],
                                'training_date': report['training_date'],
                                'samples': report['training_data']['total_samples']
                            }
                    else:
                        self.model_info[sector] = {'type': 'neural_network', 'metrics': None}
                    
                    print(f"✅ Загружена нейросетевая модель для {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки нейросети {sector}: {e}")
            
            rf_model_path = f"{self.models_dir}/model_{sector}.pkl"
            rf_scaler_path = f"{self.models_dir}/scaler_{sector}.pkl"
            
            if os.path.exists(rf_model_path) and os.path.exists(rf_scaler_path):
                try:
                    self.random_forest_models[sector] = joblib.load(rf_model_path)
                    self.random_forest_scalers[sector] = joblib.load(rf_scaler_path)
                    if sector not in self.model_info:
                        self.model_info[sector] = {'type': 'random_forest', 'metrics': None}
                    print(f"✅ Загружена Random Forest модель для {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки Random Forest {sector}: {e}")
        
        print(f"\n📊 Статус загрузки моделей:")
        for sector in sectors:
            model_type = self.model_info.get(sector, {}).get('type', 'не загружена')
            print(f"   • {sector}: {model_type}")
    
    def predict_crop_farming(self, lat, lon, shade_tolerance, optimal_temp, water_requirement, growing_days):
        """Предсказание для растениеводства"""
        
        sector = 'crop_farming'
        features = [float(lat), float(lon), float(shade_tolerance), float(optimal_temp), float(water_requirement), float(growing_days)]
        
        if sector in self.neural_models:
            X = np.array([features])
            X_scaled = self.neural_scalers[sector].transform(X)
            prediction = float(self.neural_models[sector].predict(X_scaled)[0])
            model_used = 'Нейросеть (MLPRegressor)'
        elif sector in self.random_forest_models:
            X = np.array([features])
            X_scaled = self.random_forest_scalers[sector].transform(X)
            prediction = float(self.random_forest_models[sector].predict(X_scaled)[0])
            model_used = 'Random Forest (резервная)'
        else:
            return self._fallback_prediction()
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Растениеводство',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def predict_aquaculture(self, lat, lon, water_temp, oxygen_level, stocking_density, pond_depth):
        """Предсказание для аквакультуры"""
        
        sector = 'aquaculture'
        features = [float(lat), float(lon), float(water_temp), float(oxygen_level), float(stocking_density), float(pond_depth)]
        
        if sector in self.neural_models:
            X = np.array([features])
            X_scaled = self.neural_scalers[sector].transform(X)
            prediction = float(self.neural_models[sector].predict(X_scaled)[0])
            model_used = 'Нейросеть (MLPRegressor)'
        elif sector in self.random_forest_models:
            X = np.array([features])
            X_scaled = self.random_forest_scalers[sector].transform(X)
            prediction = float(self.random_forest_models[sector].predict(X_scaled)[0])
            model_used = 'Random Forest (резервная)'
        else:
            return self._fallback_prediction()
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Аквакультура',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def predict_forestry(self, lat, lon, tree_height, canopy_density, growth_rate, wood_density):
        """Предсказание для лесного хозяйства"""
        
        sector = 'forestry'
        features = [float(lat), float(lon), float(tree_height), float(canopy_density), float(growth_rate), float(wood_density)]
        
        if sector in self.neural_models:
            X = np.array([features])
            X_scaled = self.neural_scalers[sector].transform(X)
            prediction = float(self.neural_models[sector].predict(X_scaled)[0])
            model_used = 'Нейросеть (MLPRegressor)'
        elif sector in self.random_forest_models:
            X = np.array([features])
            X_scaled = self.random_forest_scalers[sector].transform(X)
            prediction = float(self.random_forest_models[sector].predict(X_scaled)[0])
            model_used = 'Random Forest (резервная)'
        else:
            return self._fallback_prediction()
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Лесное хозяйство',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def _fallback_prediction(self):
        """Запасное предсказание"""
        return {
            'productivity_change': 100.0,
            'model_used': 'Fallback (модели не загружены)',
            'warning': 'Используется значение по умолчанию'
        }
    
    def get_model_status(self):
        """Возвращает статус всех моделей"""
        
        status = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models': {}
        }
        
        for sector in ['crop_farming', 'aquaculture', 'forestry']:
            status['models'][sector] = self.model_info.get(sector, {'type': 'не загружена'})
        
        return status

if __name__ == "__main__":
    predictor = AgrivoltaicNeuralPredictor()
    
    print("\n" + "="*70)
    print("🔮 ТЕСТИРОВАНИЕ НЕЙРОСЕТЕВЫХ ПРЕДСКАЗАНИЙ")
    print("="*70)
    
    status = predictor.get_model_status()
    print(f"\n📊 Статус моделей на {status['timestamp']}:")
    for sector, info in status['models'].items():
        print(f"   • {sector}: {info.get('type', 'неизвестно')}")
        if 'metrics' in info and info['metrics']:
            print(f"     R²: {info['metrics'].get('test_r2', '?')}, MAE: {info['metrics'].get('test_mae', '?')}")
    
    result1 = predictor.predict_crop_farming(
        lat=55.75, lon=37.62,
        shade_tolerance=0.5, optimal_temp=20,
        water_requirement=500, growing_days=120
    )
    print(f"\n🌾 Растениеводство:")
    print(f"   • Предсказание: {result1['productivity_change']:.1f}%")
    print(f"   • Модель: {result1['model_used']}")
    if 'training_samples' in result1:
        print(f"   • Обучающих примеров: {result1['training_samples']}")
    
    result2 = predictor.predict_aquaculture(
        lat=55.75, lon=37.62,
        water_temp=20, oxygen_level=7,
        stocking_density=10, pond_depth=2
    )
    print(f"\n🐟 Аквакультура:")
    print(f"   • Предсказание: {result2['productivity_change']:.1f}%")
    print(f"   • Модель: {result2['model_used']}")
    
    result3 = predictor.predict_forestry(
        lat=55.75, lon=37.62,
        tree_height=15, canopy_density=0.6,
        growth_rate=1.0, wood_density=0.6
    )
    print(f"\n🌲 Лесное хозяйство:")
    print(f"   • Предсказание: {result3['productivity_change']:.1f}%")
    print(f"   • Модель: {result3['model_used']}")