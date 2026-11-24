"""
SmartRoute Prediction & Inference Engine
Loads the trained champion pipeline and provides helper functions for single-point
and full-day curve predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'smartroute_best_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

class SmartRoutePredictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run src/train.py first.")
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.distance_lookup = self.metadata['distance_lookup']
        self.locations = self.metadata['locations']
        self.weather_types = self.metadata['weather_types']
        
    def get_route_distance(self, origin: str, destination: str) -> float:
        route_key = f"{origin} -> {destination}"
        if route_key in self.distance_lookup:
            return self.distance_lookup[route_key]
        # Reverse lookup fallback
        reverse_key = f"{destination} -> {origin}"
        if reverse_key in self.distance_lookup:
            return self.distance_lookup[reverse_key]
        return 15.0  # Safe default if unknown
        
    def predict_trip(self, origin: str, destination: str, hour: int, minute: int, day_of_week: str, weather: str) -> dict:
        distance_km = self.get_route_distance(origin, destination)
        fractional_hour = hour + (minute / 60.0)
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        is_morning_rush = 1 if 8 <= hour <= 11 else 0
        is_evening_rush = 1 if 17 <= hour <= 20 else 0
        is_rush_hour = 1 if (is_morning_rush or is_evening_rush) else 0
        
        input_df = pd.DataFrame([{
            'origin': origin,
            'destination': destination,
            'day_of_week': day_of_week,
            'weather': weather,
            'distance_km': distance_km,
            'hour': hour,
            'minute': minute,
            'fractional_hour': fractional_hour,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour
        }])
        
        predicted_duration = float(self.model.predict(input_df)[0])
        predicted_duration = max(5.0, round(predicted_duration, 1))
        
        # Calculate speed
        speed_kmh = round(distance_km / (predicted_duration / 60.0), 1)
        
        # Confidence interval (+/- 1.5 min standard error)
        lower_bound = max(5.0, round(predicted_duration - 1.5, 1))
        upper_bound = round(predicted_duration + 1.5, 1)
        
        # Traffic condition categorization
        # Free flow threshold vs rush threshold
        if speed_kmh >= 32.0:
            traffic_status = "Smooth (Fast Flow)"
            status_color = "green"
        elif speed_kmh >= 22.0:
            traffic_status = "Moderate Traffic"
            status_color = "orange"
        else:
            traffic_status = "Heavy Congestion (Peak Rush)"
            status_color = "red"
            
        return {
            'predicted_duration_minutes': predicted_duration,
            'lower_bound_minutes': lower_bound,
            'upper_bound_minutes': upper_bound,
            'distance_km': distance_km,
            'speed_kmh': speed_kmh,
            'traffic_status': traffic_status,
            'status_color': status_color,
            'is_rush_hour': bool(is_rush_hour)
        }
        
    def predict_full_day_profile(self, origin: str, destination: str, day_of_week: str, weather: str) -> pd.DataFrame:
        """
        Predicts travel duration for all 24 hours of the day (at :00 and :30 min intervals)
        to visualize the congestion wave throughout the day.
        """
        records = []
        for h in range(24):
            for m in [0, 30]:
                res = self.predict_trip(origin, destination, h, m, day_of_week, weather)
                time_label = f"{h:02d}:{m:02d}"
                records.append({
                    'time_label': time_label,
                    'hour': h,
                    'minute': m,
                    'fractional_hour': h + m/60.0,
                    'duration_minutes': res['predicted_duration_minutes'],
                    'speed_kmh': res['speed_kmh'],
                    'traffic_status': res['traffic_status']
                })
        return pd.DataFrame(records)

if __name__ == '__main__':
    predictor = SmartRoutePredictor()
    test_res = predictor.predict_trip(
        origin="UPES Bidholi",
        destination="Clock Tower",
        hour=18,
        minute=0,
        day_of_week="Friday",
        weather="Clear"
    )
    print("[SAMPLE PREDICTION - 6:00 PM Friday UPES -> Clock Tower]")
    print(json.dumps(test_res, indent=2))
