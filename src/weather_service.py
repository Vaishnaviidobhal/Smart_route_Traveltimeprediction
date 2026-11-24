"""
SmartRoute Weather Forecasting Service
Fetches real-time and future hourly weather forecasts for Dehradun (Lat: 30.3165, Lon: 78.0322)
via Open-Meteo API with offline graceful fallback.
"""

import datetime
import requests

DEHRADUN_LAT = 30.3165
DEHRADUN_LON = 78.0322

# WMO Weather Interpretation Codes to SmartRoute categories
def map_wmo_code_to_condition(code: int) -> str:
    if code in [0, 1]:
        return 'Clear'
    elif code in [2, 3]:
        return 'Cloudy'
    elif code in [45, 48]:
        return 'Fog / Haze'
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]:
        return 'Rain'
    else:
        return 'Clear'

def get_dehradun_forecast(target_date: datetime.date, target_hour: int) -> dict:
    """
    Returns the forecasted weather condition for Dehradun at the given date and hour.
    """
    target_str = f"{target_date.strftime('%Y-%m-%d')}T{target_hour:02d}:00"
    
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={DEHRADUN_LAT}&longitude={DEHRADUN_LON}&"
            f"hourly=temperature_2m,weather_code,precipitation_probability&"
            f"timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            times = data.get('hourly', {}).get('time', [])
            codes = data.get('hourly', {}).get('weather_code', [])
            temps = data.get('hourly', {}).get('temperature_2m', [])
            rain_probs = data.get('hourly', {}).get('precipitation_probability', [])
            
            if target_str in times:
                idx = times.index(target_str)
                w_code = codes[idx]
                condition = map_wmo_code_to_condition(w_code)
                return {
                    'condition': condition,
                    'temp_c': temps[idx] if idx < len(temps) else 20.0,
                    'rain_prob': rain_probs[idx] if idx < len(rain_probs) else 0,
                    'source': 'Live Satellite Forecast (Open-Meteo)',
                    'is_live': True
                }
    except Exception:
        pass
        
    # Graceful offline / out-of-range fallback based on typical Dehradun climate
    if target_hour in [5, 6, 7, 8]:
        condition = 'Fog / Haze'
        temp = 14.0
    elif 12 <= target_hour <= 16:
        condition = 'Clear'
        temp = 24.0
    else:
        condition = 'Clear'
        temp = 18.0
        
    return {
        'condition': condition,
        'temp_c': temp,
        'rain_prob': 5,
        'source': 'Climatological Model Estimate',
        'is_live': False
    }

if __name__ == '__main__':
    now = datetime.datetime.now()
    fc = get_dehradun_forecast(now.date(), 18)
    print("Forecast for today 18:00 in Dehradun:", fc)
