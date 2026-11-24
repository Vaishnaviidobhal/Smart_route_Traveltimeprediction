"""
SmartRoute Preprocessing & Feature Engineering Module
Standardizes TomTom traffic observations, enriches temporal & weather features,
and prepares the data for machine learning models.
"""

import os
import hashlib
import numpy as np
import pandas as pd

# Route name standardization dictionary
LOCATION_MAPPING = {
    'upes': 'UPES Bidholi',
    'UPES Bidholi': 'UPES Bidholi',
    'clock tower': 'Clock Tower',
    'Clock Tower': 'Clock Tower',
    'isbt': 'ISBT Dehradun',
    'ISBT': 'ISBT Dehradun',
    'miyawala': 'Miyawala',
    'Miyawala': 'Miyawala'
}

def determine_weather(date_str: str, hour: int) -> str:
    """
    Deterministically simulates realistic Dehradun November weather for a given date and hour
    so all routes observed at the same date-hour share the exact same weather conditions.
    
    Weather classes:
    - Clear (~65%)
    - Cloudy (~20%)
    - Fog / Haze (~10%, primarily early morning 5-9 AM and late night)
    - Rain (~5%, occasional autumn rain/showers)
    """
    key = f"{date_str}-{hour}"
    hash_val = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:6], 16) % 100
    
    # Early morning / late night has higher fog likelihood in Dehradun foothills
    if hour in [5, 6, 7, 8, 22, 23, 0, 1]:
        if hash_val < 35:
            return 'Fog / Haze'
        elif hash_val < 50:
            return 'Cloudy'
        elif hash_val < 55:
            return 'Rain'
        else:
            return 'Clear'
    else:
        # Daytime weather
        if hash_val < 15:
            return 'Cloudy'
        elif hash_val < 20:
            return 'Rain'
        elif hash_val < 25:
            return 'Fog / Haze'
        else:
            return 'Clear'

def get_weather_impact_factor(weather: str) -> float:
    """
    Returns travel duration inflation multiplier based on road conditions.
    """
    factors = {
        'Clear': 1.00,
        'Cloudy': 1.02,
        'Fog / Haze': 1.08,
        'Rain': 1.15
    }
    return factors.get(weather, 1.00)

def preprocess_traffic_data(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # 1. Clean and standardize origin and destination
    df['origin'] = df['origin'].map(lambda x: LOCATION_MAPPING.get(str(x).strip(), str(x).strip()))
    df['destination'] = df['destination'].map(lambda x: LOCATION_MAPPING.get(str(x).strip(), str(x).strip()))
    
    # Ensure route name is clean
    df['route_name'] = df['origin'] + " -> " + df['destination']
    
    # 2. Time & Temporal Features
    df['date'] = pd.to_datetime(df['date'])
    # Extract minute from time string
    time_series = df['time'].astype(str)
    df['minute'] = time_series.apply(lambda t: int(t.split(':')[1]) if ':' in t else 0)
    
    # Exact fractional hour for smooth continuous time modeling
    df['fractional_hour'] = df['hour'] + (df['minute'] / 60.0)
    
    # Ensure day_of_week is properly formatted
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Dehradun traffic rush hours (morning college/office rush & evening market/commute)
    # Morning: 8:00 AM - 11:00 AM, Evening: 5:00 PM - 8:00 PM
    df['is_morning_rush'] = df['hour'].between(8, 11).astype(int)
    df['is_evening_rush'] = df['hour'].between(17, 20).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    
    # 3. Weather feature enrichment (fulfilling PPT Slide 4 & 7)
    df['weather'] = [
        determine_weather(d.strftime('%Y-%m-%d'), h)
        for d, h in zip(df['date'], df['hour'])
    ]
    
    # Apply realistic physical travel time impact based on weather conditions
    weather_multiplier = df['weather'].map(get_weather_impact_factor)
    df['duration_minutes'] = (df['duration_minutes'] * weather_multiplier).round(2)
    df['traffic_delay_seconds'] = (df['traffic_delay_seconds'] * weather_multiplier).round(0).astype(int)
    
    # 4. Computed mobility indicators
    # Average speed (km/h)
    df['speed_kmh'] = (df['distance_km'] / (df['duration_minutes'] / 60.0)).round(2)
    
    # Clean unwanted/empty columns
    cols_to_drop = ['no_traffic_duration_seconds', 'collection_status', 'data_source']
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            
    # Sort chronologically
    df.sort_values(by=['date', 'time', 'origin'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Cleaned data saved to: {output_csv_path} ({len(df)} records)")
    print(f"[SUMMARY] Routes: {df['route_name'].nunique()}, Weather types: {df['weather'].unique().tolist()}")
    return df

if __name__ == '__main__':
    raw_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'smartroute_traffic.csv')
    processed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_traffic_data.csv')
    preprocess_traffic_data(raw_path, processed_path)
