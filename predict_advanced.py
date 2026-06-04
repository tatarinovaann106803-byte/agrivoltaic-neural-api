#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПРЕДСКАЗАНИЕ УРОЖАЙНОСТИ С УЧЕТОМ ТРЕХ СФЕР
"""

import joblib
import numpy as np
import os

class AgrivoltaicPredictorAdvanced:
    """Класс для предсказания с учетом сферы"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Загрузка всех трех моделей"""
        
        sectors = ['crop_farming', 'aquaculture', 'forestry']
        
        for sector in sectors:
            model_path = f"{self.models_dir}/model_{sector}.pkl"
            scaler_path = f"{self.models_dir}/scaler_{sector}.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[sector] = joblib.load(model_path)
                    self.scalers[sector] = joblib.load(scaler_path)
                    print(f"✅ Загружена модель для {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки модели {sector}: {e}")
            else:
                print(f"⚠ Модель для {sector} не найдена")
        
        print(f"\n📊 Загружено моделей: {len(self.models)} из 3")
    
    def predict_crop_farming(self, lat, lon, shade_tolerance, optimal_temp, water_requirement, growing_days):
        """Предсказание для растениеводства"""
        
        if 'crop_farming' not in self.models:
            return self._fallback_prediction()
        
        features = np.array([[
            float(lat), float(lon), float(shade_tolerance),
            float(optimal_temp), float(water_requirement), float(growing_days)
        ]])
        
        features_scaled = self.scalers['crop_farming'].transform(features)
        prediction = self.models['crop_farming'].predict(features_scaled)[0]
        
        return {
            'sector': 'crop_farming',
            'sector_name': 'Растениеводство',
            'productivity_change': float(prediction),
            'model_used': 'Random Forest (растениеводство)'
        }
    
    def predict_aquaculture(self, lat, lon, water_temp, oxygen_level, stocking_density, pond_depth):
        """Предсказание для аквакультуры"""
        
        if 'aquaculture' not in self.models:
            return self._fallback_prediction()
        
        features = np.array([[
            float(lat), float(lon), float(water_temp),
            float(oxygen_level), float(stocking_density), float(pond_depth)
        ]])
        
        features_scaled = self.scalers['aquaculture'].transform(features)
        prediction = self.models['aquaculture'].predict(features_scaled)[0]
        
        return {
            'sector': 'aquaculture',
            'sector_name': 'Аквакультура',
            'productivity_change': float(prediction),
            'model_used': 'Random Forest (аквакультура)'
        }
    
    def predict_forestry(self, lat, lon, tree_height, canopy_density, growth_rate, wood_density):
        """Предсказание для лесного хозяйства"""
        
        if 'forestry' not in self.models:
            return self._fallback_prediction()
        
        features = np.array([[
            float(lat), float(lon), float(tree_height),
            float(canopy_density), float(growth_rate), float(wood_density)
        ]])
        
        features_scaled = self.scalers['forestry'].transform(features)
        prediction = self.models['forestry'].predict(features_scaled)[0]
        
        return {
            'sector': 'forestry',
            'sector_name': 'Лесное хозяйство',
            'productivity_change': float(prediction),
            'model_used': 'Random Forest (лесное хозяйство)'
        }
    
    def predict_by_sector(self, sector, **kwargs):
        """Универсальный метод предсказания"""
        
        sector_methods = {
            'crop_farming': self.predict_crop_farming,
            'aquaculture': self.predict_aquaculture,
            'forestry': self.predict_forestry
        }
        
        if sector not in sector_methods:
            return {
                'error': f'Неизвестный сектор: {sector}',
                'available_sectors': list(sector_methods.keys())
            }
        
        method = sector_methods[sector]
        
        if sector == 'crop_farming':
            return method(
                kwargs.get('lat', 55.75), kwargs.get('lon', 37.62),
                kwargs.get('shade_tolerance', 0.5), kwargs.get('optimal_temp', 20),
                kwargs.get('water_requirement', 500), kwargs.get('growing_days', 120)
            )
        elif sector == 'aquaculture':
            return method(
                kwargs.get('lat', 55.75), kwargs.get('lon', 37.62),
                kwargs.get('water_temp', 20), kwargs.get('oxygen_level', 7),
                kwargs.get('stocking_density', 10), kwargs.get('pond_depth', 2)
            )
        else:
            return method(
                kwargs.get('lat', 55.75), kwargs.get('lon', 37.62),
                kwargs.get('tree_height', 15), kwargs.get('canopy_density', 0.6),
                kwargs.get('growth_rate', 1.0), kwargs.get('wood_density', 0.6)
            )
    
    def _fallback_prediction(self):
        """Запасное предсказание, если модель не загружена"""
        return {
            'productivity_change': 100.0,
            'model_used': 'Fallback (модель не загружена)',
            'warning': 'Используется значение по умолчанию'
        }

if __name__ == "__main__":
    predictor = AgrivoltaicPredictorAdvanced()
    
    print("\n" + "="*70)
    print("🔮 ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ")
    print("="*70)
    
    result1 = predictor.predict_crop_farming(
        lat=55.75, lon=37.62,
        shade_tolerance=0.5, optimal_temp=20,
        water_requirement=500, growing_days=120
    )
    print(f"\n🌾 Растениеводство: {result1['productivity_change']:.1f}%")
    
    result2 = predictor.predict_aquaculture(
        lat=55.75, lon=37.62,
        water_temp=20, oxygen_level=7,
        stocking_density=10, pond_depth=2
    )
    print(f"\n🐟 Аквакультура: {result2['productivity_change']:.1f}%")
    
    result3 = predictor.predict_forestry(
        lat=55.75, lon=37.62,
        tree_height=15, canopy_density=0.6,
        growth_rate=1.0, wood_density=0.6
    )
    print(f"\n🌲 Лесное хозяйство: {result3['productivity_change']:.1f}%")