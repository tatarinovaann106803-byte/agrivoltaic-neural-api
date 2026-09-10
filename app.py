#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import os
import urllib.request
import json
from datetime import datetime

app = FastAPI(title="Agrivoltaic Calculator API", version="2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ЗАГРУЗКА БАЗ ДАННЫХ ==========

def load_crop_prices():
    try:
        df = pd.read_csv('data/crop_prices.csv')
        return df
    except:
        return pd.DataFrame()

def load_faostat_yield():
    try:
        df = pd.read_csv('data/crops_yield_faostat.csv')
        if 'Element' in df.columns:
            df = df[df['Element'] == 'Yield']
        return df
    except:
        return pd.DataFrame()

def load_aquaculture_value():
    try:
        df = pd.read_csv('data/aquaculture_value.csv')
        return df
    except:
        return pd.DataFrame()

def load_forestry_value():
    try:
        df = pd.read_csv('data/forestry_trade.csv')
        return df
    except:
        return pd.DataFrame()

def load_sector_data():
    """Загрузка данных из sector_data для извлечения реальных параметров проектов"""
    sector_data = {}
    for sector in ['crop_farming', 'aquaculture', 'forestry']:
        path = f'dataset/sector_data/{sector}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            sector_data[sector] = df
        else:
            sector_data[sector] = pd.DataFrame()
    return sector_data

crop_prices_df = load_crop_prices()
faostat_yield_df = load_faostat_yield()
aquaculture_value_df = load_aquaculture_value()
forestry_value_df = load_forestry_value()
sector_data = load_sector_data()

# ========== ФУНКЦИИ ДЛЯ АВТОМАТИЧЕСКОЙ ПОДСТАНОВКИ ==========

def get_crop_price(crop_name: str, region: str = None) -> float:
    if crop_prices_df.empty:
        return 30.0
    crop_prices_df_filtered = crop_prices_df[crop_prices_df['crop'] == crop_name]
    if crop_prices_df_filtered.empty:
        return 30.0
    if region and region in crop_prices_df_filtered['region'].values:
        price = crop_prices_df_filtered[crop_prices_df_filtered['region'] == region]['price_rub_per_kg'].values[0]
        return float(price)
    else:
        return float(crop_prices_df_filtered['price_rub_per_kg'].mean())

def get_crop_yield(crop_name: str, country: str = "Russian Federation") -> float:
    if faostat_yield_df.empty:
        return 10000.0
    crop_mapping = {
        "Пшеница": "Wheat", "Кукуруза": "Maize", "Соя": "Soybeans",
        "Подсолнечник": "Sunflower seed", "Картофель": "Potatoes",
        "Сахарная свекла": "Sugar beet", "Овощи": "Vegetables"
    }
    english_name = crop_mapping.get(crop_name, crop_name)
    df_filtered = faostat_yield_df[
        (faostat_yield_df['Item'] == english_name) &
        (faostat_yield_df['Area'] == country)
    ]
    if not df_filtered.empty:
        value = df_filtered['Value'].values[0]
        if pd.notna(value):
            return float(value)
    return 10000.0

def get_aquaculture_value(species: str) -> float:
    if aquaculture_value_df.empty:
        return 150.0
    species_mapping = {
        "Карп": "Common carp", "Тилапия": "Nile tilapia",
        "Форель": "Rainbow trout", "Сом": "Catfish nei", "Осетр": "Sturgeons nei"
    }
    english_name = species_mapping.get(species, species)
    df_filtered = aquaculture_value_df[aquaculture_value_df['ASFIS species (Name)'] == english_name]
    if not df_filtered.empty:
        year_cols = [c for c in df_filtered.columns if str(c).isdigit() or (str(c).startswith('[') and str(c).endswith(']'))]
        for col in reversed(year_cols):
            try:
                value = df_filtered[col].values[0]
                if pd.notna(value) and value > 0:
                    return float(value) * 1000 / 1000
            except:
                pass
    return 150.0

def get_wood_price(wood_type: str, country: str = "Russian Federation") -> float:
    if forestry_value_df.empty:
        return 5000.0
    wood_mapping = {
        "Сосна": "Sawnwood, coniferous", "Ель": "Sawnwood, coniferous",
        "Дуб": "Sawnwood, non-coniferous", "Береза": "Sawnwood, non-coniferous",
        "Тополь": "Wood fuel, non-coniferous"
    }
    english_name = wood_mapping.get(wood_type, "Sawnwood, coniferous")
    df_filtered = forestry_value_df[
        (forestry_value_df['Item'] == english_name) &
        (forestry_value_df['Area'] == country)
    ]
    if not df_filtered.empty:
        element_col = 'Export value' if 'Export value' in df_filtered['Element'].values else 'Import value'
        df_filtered = df_filtered[df_filtered['Element'] == element_col]
        if not df_filtered.empty:
            value = df_filtered['Value'].values[0]
            if pd.notna(value) and value > 0:
                return float(value) / 1000 * 100
    return 5000.0

# ========== ФУНКЦИЯ РЕКОМЕНДАЦИЙ НА ОСНОВЕ ДАННЫХ ==========

def get_recommendations_from_data(sector: str, lat: float, crop_name: str = None):
    """
    Возвращает рекомендации по покрытию и высоте на основе реальных данных из sector_data.
    """
    df = sector_data.get(sector, pd.DataFrame())
    
    if df.empty:
        defaults = {
            'crop_farming': {'coverage': 0.32, 'height': 2.5},
            'aquaculture': {'coverage': 0.30, 'height': 2.8},
            'forestry': {'coverage': 0.25, 'height': 3.0}
        }
        return defaults.get(sector, {'coverage': 0.30, 'height': 2.5})
    
    lat_range = 5.0
    filtered = df[abs(df['latitude'] - lat) <= lat_range]
    
    if len(filtered) < 5:
        filtered = df
        if len(filtered) < 5:
            return {
                'coverage': round(df['coverage'].mean() if 'coverage' in df.columns else 0.30, 2),
                'height': round(df['height'].mean() if 'height' in df.columns else 2.5, 1)
            }
    
    has_coverage = 'coverage' in filtered.columns
    has_height = 'height' in filtered.columns
    
    coverage_adj = 0.0
    height_adj = 0.0
    
    if sector == 'crop_farming' and crop_name and 'crop_type' in filtered.columns:
        crop_filtered = filtered[filtered['crop_type'] == crop_name]
        if len(crop_filtered) > 5:
            if has_coverage:
                coverage_adj = (crop_filtered['coverage'].mean() - filtered['coverage'].mean()) * 0.5
            if has_height:
                height_adj = (crop_filtered['height'].mean() - filtered['height'].mean()) * 0.5
    
    result = {}
    if has_coverage:
        result['coverage'] = round(max(0.10, min(0.60, filtered['coverage'].mean() + coverage_adj)), 2)
    else:
        result['coverage'] = 0.30
    
    if has_height:
        result['height'] = round(max(1.5, min(5.0, filtered['height'].mean() + height_adj)), 1)
    else:
        result['height'] = 2.5
    
    return result

# ========== КЛАСС WeatherFetcher (ИСПРАВЛЕННЫЙ) ==========

class WeatherFetcher:
    def get_radiation(self, lat, lon):
        """
        Получение данных о солнечной радиации с NASA POWER API
        
        Args:
            lat (float): Широта
            lon (float): Долгота
        
        Returns:
            tuple: (годовая_радиация_kWh, список_месячных_kWh, источник)
                - годовая_радиация_kWh: в kWh/m²/year
                - список_месячных_kWh: 12 значений в kWh/m²/month
                - источник: 'NASA API' или 'Расчетный (fallback)'
        """
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            
            # Извлекаем все значения (включая среднегодовые)
            daily_values = []
            for month, value in data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].items():
                if value != -999 and value is not None:
                    daily_values.append(float(value))
            
            if daily_values:
                avg_daily_mj = np.mean(daily_values)
                # Пересчет: MJ/m²/day → kWh/m²/year
                annual_kwh = avg_daily_mj * 365 / 3.6
                
                # Проверка корректности (диапазон для kWh/m²/year)
                if 800 <= annual_kwh <= 2200:
                    # Формируем месячные значения для 2024 года
                    monthly_kwh = []
                    for m in range(1, 13):
                        key = f'2024{m:02d}'
                        if key in data['properties']['parameter']['ALLSKY_SFC_SW_DWN']:
                            value = data['properties']['parameter']['ALLSKY_SFC_SW_DWN'][key]
                            if value != -999 and value is not None:
                                # MJ/m²/day → kWh/m²/month
                                monthly_kwh.append(float(value) * 365 / 12 / 3.6)
                            else:
                                monthly_kwh.append(0.0)
                        else:
                            monthly_kwh.append(0.0)
                    
                    return round(annual_kwh, 0), monthly_kwh, 'NASA API'
        
        except Exception as e:
            print(f"⚠️ Ошибка запроса к NASA API: {e}")
        
        # ЗАПАСНОЙ ВАРИАНТ (kWh/m²/year)
        rad = 1500 - abs(lat) * 8
        rad = max(800, min(2200, rad))
        monthly = [rad / 12] * 12
        
        return round(rad, 0), monthly, 'Расчетный (fallback)'
    
    def get_radiation_annual(self, lat, lon):
        """Для обратной совместимости - возвращает только годовое значение"""
        annual, _, _ = self.get_radiation(lat, lon)
        return annual
    
    def get_temperature(self, lat, lon):
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            temps = []
            for month, value in data['properties']['parameter']['T2M'].items():
                if value != -999 and value is not None:
                    temps.append(float(value))
            if temps:
                return round(np.mean(temps), 1)
        except:
            pass
        return 15.0

# ========== КЛАСС Calculator (ИСПРАВЛЕННЫЙ) ==========

class Calculator:
    def __init__(self):
        self.panel_width = 2.134      # метра
        self.panel_height = 1.051     # метра
        self.panel_power = 0.445      # кВт (445 Вт)
        self.panel_area = self.panel_width * self.panel_height  # м²
        self.panel_efficiency = 0.24  # 24% (современные панели)
        
        # Потери эффективности
        self.soiling_loss = 0.97      # загрязнение
        self.thermal_loss = 0.94      # нагрев
        self.inverter_loss = 0.97     # инвертор
        self.cable_loss = 0.98        # кабели
        self.mismatch_loss = 0.99     # несоответствие
        
        self.total_efficiency = (self.soiling_loss * self.thermal_loss * 
                                 self.inverter_loss * self.cable_loss * self.mismatch_loss)
    
    def monthly_radiation(self, lat, annual_radiation):
        """
        Разбивает годовую радиацию на месячные значения с учетом широты
        
        Args:
            lat (float): Широта
            annual_radiation (float): Годовая радиация в kWh/m²
        
        Returns:
            np.array: Массив из 12 месячных значений в kWh/m²/month
        """
        months = np.arange(1, 13)
        
        # Склонение солнца по месяцам
        declination = -23.45 * np.cos(2 * np.pi * (months - 1) / 12)
        
        # Высота солнца
        sun_alt = 90 - np.abs(lat - declination)
        sun_alt = np.clip(sun_alt, 10, 90)
        
        # Относительный фактор для каждого месяца
        monthly_factor = np.sin(np.radians(sun_alt)) / np.sin(np.radians(90 - np.abs(lat)))
        monthly_factor = monthly_factor / monthly_factor.sum() * 12
        
        return annual_radiation / 12 * monthly_factor
    
    def solar_energy_pvsyst(self, area_ha, coverage, radiation, lat):
        """
        Расчет солнечной энергии с учетом сезонности
        
        Args:
            area_ha (float): Площадь в гектарах
            coverage (float): Коэффициент покрытия панелями (0-1)
            radiation (float): Годовая радиация в kWh/m²/year
            lat (float): Широта
        
        Returns:
            dict: Результаты расчета
        """
        area_m2 = area_ha * 10000
        panel_area_total = area_m2 * coverage
        
        # Количество панелей
        num_panels = int(panel_area_total / self.panel_area)
        if num_panels < 1:
            num_panels = 1
        total_power = num_panels * self.panel_power  # кВт
        
        # Оптимальный угол наклона
        tilt_optimal = abs(lat) * 0.9 + 5
        tilt_optimal = min(55, max(20, tilt_optimal))
        tilt_factor = np.cos(np.radians(tilt_optimal - abs(lat))) * 0.95 + 0.05
        
        # Получаем месячное распределение радиации
        monthly_rad = self.monthly_radiation(lat, radiation)
        
        # Расчет месячной энергии
        monthly_energy = (panel_area_total * monthly_rad * 
                         self.total_efficiency * self.panel_efficiency * tilt_factor)
        
        annual_energy = monthly_energy.sum()
        specific_yield = annual_energy / total_power if total_power > 0 else 0
        
        return {
            'num_panels': num_panels,
            'total_power': total_power,
            'annual_energy': annual_energy,
            'monthly_energy': monthly_energy.tolist(),  # список из 12 значений
            'tilt_angle': tilt_optimal,
            'specific_yield': specific_yield,
            'total_efficiency': self.total_efficiency,
            'panel_area_total': panel_area_total
        }
    
    def economics(self, energy, product_income, energy_price, capex_per_kw=60000):
        energy_income = energy['annual_energy'] * energy_price
        total_income = energy_income + product_income
        capex = energy['total_power'] * capex_per_kw
        opex = capex * 0.015
        net_income = total_income - opex
        roi = capex / net_income if net_income > 0 else 999
        return {
            'energy_income': energy_income,
            'total_income': total_income,
            'capex': capex,
            'net_income': net_income,
            'roi_years': roi
        }

# ========== ЗАГРУЗКА AI МОДЕЛЕЙ ==========

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        model_files = {
            'crop': ('models/model_crop_farming.pkl', 'models/scaler_crop_farming.pkl'),
            'aqua': ('models/model_aquaculture.pkl', 'models/scaler_aquaculture.pkl'),
            'forest': ('models/model_forestry.pkl', 'models/scaler_forestry.pkl')
        }
        for sector, (model_path, scaler_path) in model_files.items():
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[sector] = joblib.load(model_path)
                    self.scalers[sector] = joblib.load(scaler_path)
                    print(f"Загружена модель {sector}")
                except Exception as e:
                    print(f"Ошибка загрузки {sector}: {e}")
    
    def predict(self, sector, features):
        if sector not in self.models:
            fallbacks = {'crop': 85, 'aqua': 8, 'forest': 12}
            return fallbacks.get(sector, 50)
        try:
            X = np.array([features])
            X_scaled = self.scalers[sector].transform(X)
            return float(self.models[sector].predict(X_scaled)[0])
        except:
            fallbacks = {'crop': 85, 'aqua': 8, 'forest': 12}
            return fallbacks.get(sector, 50)

weather_fetcher = WeatherFetcher()
calculator = Calculator()
predictor = ModelPredictor()

# ========== PYDANTIC МОДЕЛИ ==========

class CalculationRequest(BaseModel):
    sector: str
    lat: float
    lon: float
    area_ha: float
    coverage: float
    height: float
    energy_price: float
    temp: Optional[float] = None
    region: Optional[str] = None
    country: Optional[str] = "Russian Federation"
    
    crop_price: Optional[float] = None
    base_yield: Optional[float] = None
    crop_name: Optional[str] = "Пшеница"
    
    fish_price: Optional[float] = None
    stocking_density: Optional[float] = 10
    oxygen_level: Optional[float] = 7
    pond_depth: Optional[float] = 3
    fish_name: Optional[str] = "Карп"
    
    wood_price: Optional[float] = None
    tree_height: Optional[float] = 15
    canopy_density: Optional[float] = 0.6
    forest_name: Optional[str] = "Сосна"

# ========== API ENDPOINTS ==========

@app.get("/")
def root():
    return {"service": "Agrivoltaic Calculator API", "version": "2.7", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiation")
def get_radiation_endpoint(lat: float, lon: float):
    """
    Возвращает годовую радиацию и месячное распределение
    """
    annual, monthly, source = weather_fetcher.get_radiation(lat, lon)
    return {
        "radiation_annual": annual,
        "radiation_monthly": monthly,
        "source": source,
        "lat": lat,
        "lon": lon
    }

@app.post("/recommendations")
def get_recommendations(request: CalculationRequest):
    """
    Возвращает рекомендации по покрытию и высоте на основе реальных данных из sector_data
    """
    try:
        crop_name = request.crop_name if request.sector == "crop" else None
        rec = get_recommendations_from_data(request.sector, request.lat, crop_name)
        
        msg = "Рекомендации подобраны на основе реальных данных из агривольтаических проектов."
        if request.sector == "crop" and crop_name:
            msg += f" Для культуры {crop_name} использованы параметры проектов с аналогичной широтой."
        
        return {
            "success": True,
            "recommendations": {
                "coverage": rec.get("coverage", 0.30),
                "height": rec.get("height", 2.5),
                "message": msg,
                "data_source": "реальные проекты из sector_data"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== ОСНОВНОЙ ЭНДПОИНТ /calculate ==========

@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        # Получаем радиацию (годовую и месячную)
        radiation_annual, radiation_monthly, radiation_source = weather_fetcher.get_radiation(request.lat, request.lon)
        
        auto_temp = weather_fetcher.get_temperature(request.lat, request.lon)
        temp = request.temp if request.temp is not None else auto_temp
        
        # Энергетический расчет с сезонностью
        energy = calculator.solar_energy_pvsyst(
            request.area_ha, request.coverage, radiation_annual, request.lat
        )
        
        # ========== РАСТЕНИЕВОДСТВО ==========
        if request.sector == "crop":
            crop_params = {
                "Пшеница": {"shade_tolerance": 0.35, "optimal_temp": 18, "water_requirement": 450, "growing_days": 120},
                "Кукуруза": {"shade_tolerance": 0.30, "optimal_temp": 22, "water_requirement": 550, "growing_days": 130},
                "Соя": {"shade_tolerance": 0.40, "optimal_temp": 20, "water_requirement": 500, "growing_days": 125},
                "Подсолнечник": {"shade_tolerance": 0.45, "optimal_temp": 21, "water_requirement": 480, "growing_days": 110},
                "Картофель": {"shade_tolerance": 0.50, "optimal_temp": 17, "water_requirement": 400, "growing_days": 100},
                "Сахарная свекла": {"shade_tolerance": 0.55, "optimal_temp": 19, "water_requirement": 520, "growing_days": 140},
                "Овощи": {"shade_tolerance": 0.60, "optimal_temp": 20, "water_requirement": 450, "growing_days": 90}
            }
            
            params = crop_params.get(request.crop_name, crop_params["Пшеница"])
            
            features = [
                request.lat, request.lon,
                params["shade_tolerance"], params["optimal_temp"],
                params["water_requirement"], params["growing_days"]
            ]
            
            productivity_change = predictor.predict('crop', features)
            
            if productivity_change < 50:
                productivity_change = 50.0
            if productivity_change > 150:
                productivity_change = 150.0
            
            crop_price = request.crop_price if request.crop_price is not None else get_crop_price(request.crop_name, request.region)
            base_yield = request.base_yield if request.base_yield is not None else get_crop_yield(request.crop_name, request.country)
            
            product_income = base_yield * request.area_ha * (productivity_change / 100) * crop_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            
            result = {
                "sector": "crop", "sector_name": "Растениеводство", "culture": request.crop_name,
                "temperature_used": temp, "region": request.region,
                "crop_price_used": crop_price, "base_yield_used": base_yield,
                "productivity": {"value": productivity_change, "unit": "%", "label": "изменение урожайности"},
                "energy": energy, "economics": economics
            }
        
        # ========== АКВАКУЛЬТУРА ==========
        elif request.sector == "aqua":
            features = [request.lat, request.lon, temp, request.oxygen_level, request.stocking_density, request.pond_depth]
            productivity = predictor.predict('aqua', features)
            fish_price = request.fish_price if request.fish_price is not None else get_aquaculture_value(request.fish_name)
            product_income = productivity * request.area_ha * fish_price * 1000
            economics = calculator.economics(energy, product_income, request.energy_price, capex_per_kw=70000)
            result = {
                "sector": "aqua", "sector_name": "Аквакультура", "culture": request.fish_name,
                "temperature_used": temp, "region": request.region,
                "fish_price_used": fish_price,
                "productivity": {"value": productivity, "unit": "т/га", "label": "продуктивность"},
                "energy": energy, "economics": economics
            }
        
        # ========== ЛЕСНОЕ ХОЗЯЙСТВО ==========
        else:
            features = [request.lat, request.lon, request.tree_height, request.canopy_density, 1.0, 0.6]
            productivity = predictor.predict('forest', features)
            wood_price = request.wood_price if request.wood_price is not None else get_wood_price(request.forest_name, request.country)
            product_income = productivity * request.area_ha * wood_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            result = {
                "sector": "forest", "sector_name": "Лесное хозяйство", "culture": request.forest_name,
                "temperature_used": temp, "region": request.region,
                "wood_price_used": wood_price,
                "productivity": {"value": productivity, "unit": "м³/га/год", "label": "прирост древесины"},
                "energy": energy, "economics": economics
            }
        
        # Добавляем информацию о радиации
        result["radiation"] = radiation_annual
        result["radiation_monthly"] = radiation_monthly
        result["radiation_source"] = radiation_source
        result["location"] = {"lat": request.lat, "lon": request.lon}
        result["area_ha"] = request.area_ha
        result["coverage"] = request.coverage
        result["height"] = request.height
        
        return {"success": True, "data": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
