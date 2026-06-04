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

app = FastAPI(title="Agrivoltaic Calculator API", version="2.4")

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
    """Стоимость аквакультуры из FAO FishStatJ (USD 1000)"""
    try:
        df = pd.read_csv('data/aquaculture_value.csv')
        return df
    except:
        return pd.DataFrame()

def load_aquaculture_production():
    """Производство аквакультуры из FAO FishStatJ (тонны)"""
    try:
        df = pd.read_csv('data/aquaculture_production.csv')
        return df
    except:
        return pd.DataFrame()

def load_forestry_value():
    try:
        df = pd.read_csv('data/forestry_trade.csv')
        return df
    except:
        return pd.DataFrame()

# Загружаем все базы
crop_prices_df = load_crop_prices()
faostat_yield_df = load_faostat_yield()
aquaculture_value_df = load_aquaculture_value()
aquaculture_production_df = load_aquaculture_production()
forestry_value_df = load_forestry_value()

# ========== ФУНКЦИИ ДЛЯ АВТОМАТИЧЕСКОЙ ПОДСТАНОВКИ ==========

# 1. РАСТЕНИЕВОДСТВО

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
        avg_price = crop_prices_df_filtered['price_rub_per_kg'].mean()
        return float(avg_price)

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


# 2. АКВАКУЛЬТУРА

def get_aquaculture_value(species: str, environment: str = "Freshwater") -> float:
    """
    Получение стоимости аквакультуры (руб/кг)
    Источник: FAO FishStatJ (Value USD 1000)
    """
    if aquaculture_value_df.empty:
        return 150.0
    
    # Сопоставление русских названий с английскими (ASFIS)
    species_mapping = {
        "Карп": "Common carp",
        "Тилапия": "Nile tilapia",
        "Форель": "Rainbow trout",
        "Сом": "Catfish nei",
        "Осетр": "Sturgeons nei",
        "Лосось": "Atlantic salmon",
        "Креветка": "Penaeus shrimps",
        "Мидия": "Mussels",
        "Устрица": "Oysters"
    }
    
    english_name = species_mapping.get(species, species)
    
    # Фильтруем по виду и среде выращивания
    df_filtered = aquaculture_value_df[
        (aquaculture_value_df['ASFIS species (Name)'] == english_name) &
        (aquaculture_value_df['Environment (Name)'] == environment)
    ]
    
    if df_filtered.empty:
        # Пробуем без фильтра по среде
        df_filtered = aquaculture_value_df[aquaculture_value_df['ASFIS species (Name)'] == english_name]
    
    if not df_filtered.empty:
        # Находим столбцы с годами
        year_cols = [c for c in df_filtered.columns if str(c).isdigit() or (str(c).startswith('[') and str(c).endswith(']'))]
        year_cols_clean = []
        for c in year_cols:
            try:
                year = int(str(c).strip('[]'))
                year_cols_clean.append((year, c))
            except:
                pass
        
        if year_cols_clean:
            # Берём последний доступный год (максимальный)
            year_cols_clean.sort(reverse=True)
            for year, col in year_cols_clean:
                value = df_filtered[col].values[0]
                if pd.notna(value) and value > 0:
                    # Конвертация: из 1000 USD в рубли за кг
                    # 1000 USD = ~100000 руб (курс 100)
                    # value уже в тысячах USD, значит value * 1000 = USD
                    usd_value = float(value) * 1000
                    rub_value = usd_value * 100  # курс 100 руб за USD
                    # Возвращаем цену за кг (условно делим на 1000)
                    return rub_value / 1000
    
    return 150.0

def get_aquaculture_production(species: str, country: str = "Russian Federation", environment: str = "Freshwater") -> float:
    """
    Получение объёма производства аквакультуры (тонны)
    Источник: FAO FishStatJ (Production)
    """
    if aquaculture_production_df.empty:
        return 10.0
    
    species_mapping = {
        "Карп": "Common carp", "Тилапия": "Nile tilapia",
        "Форель": "Rainbow trout", "Сом": "Catfish nei", "Осетр": "Sturgeons nei"
    }
    english_name = species_mapping.get(species, species)
    
    df_filtered = aquaculture_production_df[
        (aquaculture_production_df['ASFIS species (Name)'] == english_name) &
        (aquaculture_production_df['Country (Name)'] == country) &
        (aquaculture_production_df['Environment (Name)'] == environment)
    ]
    
    if df_filtered.empty:
        df_filtered = aquaculture_production_df[
            (aquaculture_production_df['ASFIS species (Name)'] == english_name) &
            (aquaculture_production_df['Country (Name)'] == country)
        ]
    
    if not df_filtered.empty:
        year_cols = [c for c in df_filtered.columns if str(c).isdigit() or (str(c).startswith('[') and str(c).endswith(']'))]
        for col in reversed(year_cols):
            try:
                value = df_filtered[col].values[0]
                if pd.notna(value) and value > 0:
                    return float(value)
            except:
                pass
    
    return 10.0


# 3. ЛЕСНОЕ ХОЗЯЙСТВО

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


# ========== КЛАССЫ WeatherFetcher, Calculator, ModelPredictor ==========

class WeatherFetcher:
    def get_radiation(self, lat, lon):
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            daily_values = []
            for month, value in data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].items():
                if value != -999 and value is not None:
                    daily_values.append(float(value))
            if daily_values:
                avg_daily = np.mean(daily_values)
                annual = avg_daily * 365
                if 800 <= annual <= 2200:
                    return round(annual, 0)
        except Exception as e:
            print(f"Ошибка радиации: {e}")
        rad = 1500 - abs(lat) * 8
        return round(max(800, min(2200, rad)), 0)
    
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
                annual = np.mean(temps)
                return round(annual, 1)
        except Exception as e:
            print(f"Ошибка температуры: {e}")
        return 15.0

class Calculator:
    def __init__(self):
        self.panel_width = 2.134
        self.panel_height = 1.051
        self.panel_power = 0.445
        self.panel_area = self.panel_width * self.panel_height
        self.panel_efficiency = 0.20
        
    def solar_energy_pvsyst(self, area_ha, coverage, radiation, lat):
        area_m2 = area_ha * 10000
        panel_area_total = area_m2 * coverage
        num_panels = int(panel_area_total / self.panel_area)
        total_power = num_panels * self.panel_power
        tilt_optimal = abs(lat) * 0.9 + 5
        tilt_optimal = min(55, max(20, tilt_optimal))
        tilt_factor = np.cos(np.radians(tilt_optimal - abs(lat))) * 0.95 + 0.05
        soiling_loss = 0.97
        thermal_loss = 0.92
        inverter_loss = 0.97
        cable_loss = 0.98
        mismatch_loss = 0.99
        total_efficiency = soiling_loss * thermal_loss * inverter_loss * cable_loss * mismatch_loss
        annual_energy = panel_area_total * radiation * total_efficiency * self.panel_efficiency * tilt_factor
        specific_yield = annual_energy / total_power if total_power > 0 else 0
        monthly_energy = [annual_energy / 12] * 12
        return {
            'num_panels': num_panels,
            'total_power': total_power,
            'annual_energy': annual_energy,
            'monthly_energy': monthly_energy,
            'tilt_angle': tilt_optimal,
            'specific_yield': specific_yield,
            'total_efficiency': total_efficiency
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
    
    # Растениеводство
    crop_price: Optional[float] = None
    base_yield: Optional[float] = None
    crop_name: Optional[str] = "Пшеница"
    
    # Аквакультура
    fish_price: Optional[float] = None
    stocking_density: Optional[float] = 10
    oxygen_level: Optional[float] = 7
    pond_depth: Optional[float] = 3
    fish_name: Optional[str] = "Карп"
    
    # Лесное хозяйство
    wood_price: Optional[float] = None
    tree_height: Optional[float] = 15
    canopy_density: Optional[float] = 0.6
    forest_name: Optional[str] = "Сосна"

# ========== API ENDPOINTS ==========

@app.get("/")
def root():
    return {"service": "Agrivoltaic Calculator API", "version": "2.4", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiation")
def get_radiation(lat: float, lon: float):
    radiation = weather_fetcher.get_radiation(lat, lon)
    return {"radiation": radiation, "lat": lat, "lon": lon}

@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        radiation = weather_fetcher.get_radiation(request.lat, request.lon)
        auto_temp = weather_fetcher.get_temperature(request.lat, request.lon)
        temp = request.temp if request.temp is not None else auto_temp
        
        energy = calculator.solar_energy_pvsyst(
            request.area_ha, request.coverage, radiation, request.lat
        )
        
        # ========== РАСТЕНИЕВОДСТВО ==========
        if request.sector == "crop":
            features = [request.lat, request.lon, 0.5, temp, 500.0, 120.0]
            productivity_change = predictor.predict('crop', features)
            
            crop_price = request.crop_price if request.crop_price is not None else get_crop_price(request.crop_name, request.region)
            base_yield = request.base_yield if request.base_yield is not None else get_crop_yield(request.crop_name, request.country)
            
            product_income = base_yield * request.area_ha * (productivity_change / 100) * crop_price
            base_income = base_yield * request.area_ha * crop_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            
            result = {
                "sector": "crop", "sector_name": "Растениеводство", "culture": request.crop_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "region": request.region, "country": request.country,
                "crop_price_used": crop_price, "crop_price_source": "авто" if request.crop_price is None else "пользователь",
                "base_yield_used": base_yield, "base_yield_source": "FAO" if request.base_yield is None else "пользователь",
                "productivity": {"value": productivity_change, "unit": "%", "label": "изменение урожайности"},
                "energy": energy, "economics": economics
            }
        
        # ========== АКВАКУЛЬТУРА ==========
        elif request.sector == "aqua":
            features = [request.lat, request.lon, temp, request.oxygen_level, request.stocking_density, request.pond_depth]
            productivity = predictor.predict('aqua', features)
            
            fish_price = request.fish_price if request.fish_price is not None else get_aquaculture_value(request.fish_name)
            
            product_income = productivity * request.area_ha * fish_price * 1000
            base_income = request.stocking_density * request.area_ha * fish_price * 1000
            economics = calculator.economics(energy, product_income, request.energy_price, capex_per_kw=70000)
            
            result = {
                "sector": "aqua", "sector_name": "Аквакультура", "culture": request.fish_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "region": request.region,
                "fish_price_used": fish_price, "fish_price_source": "FAO" if request.fish_price is None else "пользователь",
                "productivity": {"value": productivity, "unit": "т/га", "label": "продуктивность"},
                "energy": energy, "economics": economics
            }
        
        # ========== ЛЕСНОЕ ХОЗЯЙСТВО ==========
        else:
            features = [request.lat, request.lon, request.tree_height, request.canopy_density, 1.0, 0.6]
            productivity = predictor.predict('forest', features)
            
            wood_price = request.wood_price if request.wood_price is not None else get_wood_price(request.forest_name, request.country)
            
            product_income = productivity * request.area_ha * wood_price
            base_income = 8 * request.area_ha * wood_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            
            result = {
                "sector": "forest", "sector_name": "Лесное хозяйство", "culture": request.forest_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "region": request.region, "country": request.country,
                "wood_price_used": wood_price, "wood_price_source": "FAO" if request.wood_price is None else "пользователь",
                "productivity": {"value": productivity, "unit": "м³/га/год", "label": "прирост древесины"},
                "energy": energy, "economics": economics
            }
        
        result["radiation"] = radiation
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
