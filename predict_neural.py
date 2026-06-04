#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПРЕДСКАЗАНИЕ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ
С автоматическим выбором лучшей доступной модели
"""

import joblib
import numpy as np
import os
import json
from datetime import datetime

class AgrivoltaicNeuralPredictor:
    """Класс для предсказания с автоматическим выбором модели"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.neural_models = {}
        self.neural_scalers = {}
        self.random_forest_models = {}
        self.random_forest_scalers = {}
        self.model_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Загрузка всех доступных моделей (нейросетевых и Random Forest)"""
        
        sectors = ['crop_farming', 'aquaculture', 'forestry']
        
        for sector in sectors:
            # Загрузка нейросетевой модели
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
                                'metrics': report.get('metrics', {}),
                                'training_date': report.get('training_date', 'неизвестно'),
                                'samples': report.get('training_data', {}).get('total_samples', 0)
                            }
                    else:
                        self.model_info[sector] = {'type': 'neural_network', 'metrics': None}
                    
                    print(f"✅ Загружена нейросетевая модель для {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки нейросети {sector}: {e}")
            
            # Загрузка Random Forest модели
            rf_model_path = f"{self.models_dir}/model_{sector}.pkl"
            rf_scaler_path = f"{self.models_dir}/scaler_{sector}.pkl"
            
            if os.path.exists(rf_model_path) and os.path.exists(rf_scaler_path):
                try:
                    self.random_forest_models[sector] = joblib.load(rf_model_path)
                    self.random_forest_scalers[sector] = joblib.load(rf_scaler_path)
                    
                    if sector not in self.model_info:
                        # Пытаемся найти отчёт
                        report_path = f"{self.models_dir}/smart_report_{sector}.json"
                        if os.path.exists(report_path):
                            with open(report_path, 'r', encoding='utf-8') as f:
                                report = json.load(f)
                                self.model_info[sector] = {
                                    'type': report.get('model_used', 'random_forest'),
                                    'metrics': report.get('metrics', {}),
                                    'training_date': report.get('training_date', 'неизвестно'),
                                    'samples': report.get('data_size', 0)
                                }
                        else:
                            self.model_info[sector] = {'type': 'random_forest', 'metrics': None}
                    
                    print(f"✅ Загружена Random Forest модель для {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки Random Forest {sector}: {e}")
        
        print(f"\n📊 Статус загрузки моделей:")
        for sector in sectors:
            if sector in self.model_info:
                info = self.model_info[sector]
                print(f"   • {sector}: {info.get('type', '?')} | samples={info.get('samples', '?')}")
            else:
                print(f"   • {sector}: не загружена")
    
    def _get_best_model(self, sector):
        """Возвращает лучшую доступную модель для сектора"""
        
        # Приоритет: нейросеть → Random Forest → Fallback
        if sector in self.neural_models:
            return {
                'model': self.neural_models[sector],
                'scaler': self.neural_scalers[sector],
                'type': 'neural_network'
            }
        elif sector in self.random_forest_models:
            return {
                'model': self.random_forest_models[sector],
                'scaler': self.random_forest_scalers[sector],
                'type': 'random_forest'
            }
        else:
            return None
    
    def predict_crop_farming(self, lat, lon, shade_tolerance, optimal_temp, water_requirement, growing_days):
        """Предсказание для растениеводства"""
        
        sector = 'crop_farming'
        features = [float(lat), float(lon), float(shade_tolerance), float(optimal_temp), float(water_requirement), float(growing_days)]
        
        best = self._get_best_model(sector)
        
        if best:
            X = np.array([features])
            X_scaled = best['scaler'].transform(X)
            prediction = float(best['model'].predict(X_scaled)[0])
            model_used = 'Нейросеть' if best['type'] == 'neural_network' else 'Random Forest'
        else:
            return self._fallback_prediction(sector)
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Растениеводство',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_type': best['type'],
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def predict_aquaculture(self, lat, lon, water_temp, oxygen_level, stocking_density, pond_depth):
        """Предсказание для аквакультуры"""
        
        sector = 'aquaculture'
        features = [float(lat), float(lon), float(water_temp), float(oxygen_level), float(stocking_density), float(pond_depth)]
        
        best = self._get_best_model(sector)
        
        if best:
            X = np.array([features])
            X_scaled = best['scaler'].transform(X)
            prediction = float(best['model'].predict(X_scaled)[0])
            model_used = 'Нейросеть' if best['type'] == 'neural_network' else 'Random Forest'
        else:
            return self._fallback_prediction(sector)
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Аквакультура',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_type': best['type'],
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def predict_forestry(self, lat, lon, tree_height, canopy_density, growth_rate, wood_density):
        """Предсказание для лесного хозяйства"""
        
        sector = 'forestry'
        features = [float(lat), float(lon), float(tree_height), float(canopy_density), float(growth_rate), float(wood_density)]
        
        best = self._get_best_model(sector)
        
        if best:
            X = np.array([features])
            X_scaled = best['scaler'].transform(X)
            prediction = float(best['model'].predict(X_scaled)[0])
            model_used = 'Нейросеть' if best['type'] == 'neural_network' else 'Random Forest'
        else:
            return self._fallback_prediction(sector)
        
        model_info = self.model_info.get(sector, {})
        
        return {
            'sector': sector,
            'sector_name': 'Лесное хозяйство',
            'productivity_change': prediction,
            'model_used': model_used,
            'model_type': best['type'],
            'model_quality': model_info.get('metrics', {}),
            'training_samples': model_info.get('samples', 0),
            'training_date': model_info.get('training_date', 'неизвестно')
        }
    
    def predict_by_sector(self, sector, **kwargs):
        """Универсальный метод предсказания по названию сектора"""
        
        if sector == 'crop_farming':
            return self.predict_crop_farming(
                kwargs.get('lat', 55.75),
                kwargs.get('lon', 37.62),
                kwargs.get('shade_tolerance', 0.5),
                kwargs.get('optimal_temp', 20),
                kwargs.get('water_requirement', 500),
                kwargs.get('growing_days', 120)
            )
        elif sector == 'aquaculture':
            return self.predict_aquaculture(
                kwargs.get('lat', 55.75),
                kwargs.get('lon', 37.62),
                kwargs.get('water_temp', 20),
                kwargs.get('oxygen_level', 7),
                kwargs.get('stocking_density', 10),
                kwargs.get('pond_depth', 2)
            )
        elif sector == 'forestry':
            return self.predict_forestry(
                kwargs.get('lat', 55.75),
                kwargs.get('lon', 37.62),
                kwargs.get('tree_height', 15),
                kwargs.get('canopy_density', 0.6),
                kwargs.get('growth_rate', 1.0),
                kwargs.get('wood_density', 0.6)
            )
        else:
            return {
                'error': f'Неизвестный сектор: {sector}',
                'available_sectors': ['crop_farming', 'aquaculture', 'forestry']
            }
    
    def _fallback_prediction(self, sector):
        """Запасное предсказание, если модели не загружены"""
        
        fallback_values = {
            'crop_farming': 85.0,
            'aquaculture': 8.0,
            'forestry': 12.0
        }
        
        return {
            'sector': sector,
            'productivity_change': fallback_values.get(sector, 50.0),
            'model_used': 'Fallback (модели не загружены)',
            'model_type': 'fallback',
            'warning': 'Используется значение по умолчанию'
        }
    
    def get_model_status(self):
        """Возвращает статус всех моделей"""
        
        status = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models': {}
        }
        
        for sector in ['crop_farming', 'aquaculture', 'forestry']:
            if sector in self.model_info:
                info = self.model_info[sector]
                status['models'][sector] = {
                    'type': info.get('type', 'неизвестно'),
                    'training_samples': info.get('samples', 0),
                    'training_date': info.get('training_date', 'неизвестно'),
                    'metrics': info.get('metrics', {})
                }
            else:
                # Проверяем наличие моделей напрямую
                has_neural = sector in self.neural_models
                has_rf = sector in self.random_forest_models
                if has_neural:
                    status['models'][sector] = {'type': 'neural_network_available', 'training_samples': '?'}
                elif has_rf:
                    status['models'][sector] = {'type': 'random_forest_available', 'training_samples': '?'}
                else:
                    status['models'][sector] = {'type': 'not_loaded'}
        
        return status

# Для тестирования
if __name__ == "__main__":
    predictor = AgrivoltaicNeuralPredictor()
    
    print("\n" + "="*70)
    print("🔮 ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ")
    print("="*70)
    
    status = predictor.get_model_status()
    print(f"\n📊 Статус моделей на {status['timestamp']}:")
    for sector, info in status['models'].items():
        print(f"   • {sector}: {info.get('type', 'неизвестно')}")
        if 'metrics' in info and info['metrics']:
            print(f"     R²: {info['metrics'].get('test_r2', '?')}, MAE: {info['metrics'].get('test_mae', '?')}")
    
    # Тест для растениеводства
    result1 = predictor.predict_crop_farming(
        lat=55.75, lon=37.62,
        shade_tolerance=0.5, optimal_temp=20,
        water_requirement=500, growing_days=120
    )
    print(f"\n🌾 Растениеводство:")
    print(f"   • Предсказание: {result1['productivity_change']:.1f}%")
    print(f"   • Модель: {result1['model_used']} ({result1.get('model_type', '?')})")
    if 'training_samples' in result1:
        print(f"   • Обучающих примеров: {result1['training_samples']}")
    
    # Тест для аквакультуры
    result2 = predictor.predict_aquaculture(
        lat=55.75, lon=37.62,
        water_temp=20, oxygen_level=7,
        stocking_density=10, pond_depth=2
    )
    print(f"\n🐟 Аквакультура:")
    print(f"   • Предсказание: {result2['productivity_change']:.1f}%")
    print(f"   • Модель: {result2['model_used']} ({result2.get('model_type', '?')})")
    
    # Тест для лесного хозяйства
    result3 = predictor.predict_forestry(
        lat=55.75, lon=37.62,
        tree_height=15, canopy_density=0.6,
        growth_rate=1.0, wood_density=0.6
    )
    print(f"\n🌲 Лесное хозяйство:")
    print(f"   • Предсказание: {result3['productivity_change']:.1f}%")
    print(f"   • Модель: {result3['model_used']} ({result3.get('model_type', '?')})")
