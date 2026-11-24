"""
SmartRoute - AI Travel Time Prediction for Selected Routes
Interactive GUI Built with Streamlit
Designed for B.Tech AIML Semester Project Submission
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import SmartRoutePredictor
from src.weather_service import get_dehradun_forecast

# Page Configuration
st.set_page_config(
    page_title="SmartRoute | Travel Time Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Professional Polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .badge-green { background-color: #DEF7EC; color: #03543F; }
    .badge-orange { background-color: #FEF08A; color: #713F12; }
    .badge-red { background-color: #FDE8E8; color: #9B1C1C; }
</style>
""", unsafe_allow_html=True)

# Landmark Coordinates for Map Visualization
COORDINATES = {
    'UPES Bidholi': {'lat': 30.4158, 'lon': 77.9659},
    'Clock Tower': {'lat': 30.3256, 'lon': 78.0437},
    'ISBT Dehradun': {'lat': 30.2863, 'lon': 78.0076},
    'Miyawala': {'lat': 30.2705, 'lon': 78.1066}
}

@st.cache_resource
def load_predictor():
    return SmartRoutePredictor()

predictor = load_predictor()

# -------------------------------------------------------------
# SIDEBAR - INPUT CONTROLS
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/route.png", width=64)
st.sidebar.title("SmartRoute Controls")
st.sidebar.markdown("**Predictive Traffic & Journey Estimator**")
st.sidebar.markdown("---")

all_locations = predictor.locations
origin = st.sidebar.selectbox("🚩 Origin Point", all_locations, index=0)

destination_options = [loc for loc in all_locations if loc != origin]
destination = st.sidebar.selectbox("🏁 Destination Point", destination_options, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Schedule & Departure")

# Date Picker (defaults to today)
selected_date = st.sidebar.date_input("Departure Date", datetime.date.today())
selected_day = selected_date.strftime("%A")
st.sidebar.caption(f"Day of Week: **{selected_day}**")

# Time Selection
departure_time = st.sidebar.time_input("Departure Time", datetime.time(18, 0))
hour = departure_time.hour
minute = departure_time.minute

# Quick Time Presets
st.sidebar.markdown("**Quick Rush Presets:**")
preset_col1, preset_col2 = st.sidebar.columns(2)
if preset_col1.button("🌅 09:00 AM (Rush)"):
    hour, minute = 9, 0
if preset_col2.button("🌆 06:00 PM (Rush)"):
    hour, minute = 18, 0

st.sidebar.markdown("---")
st.sidebar.subheader("⛅ Weather Intelligence")

# Live Weather Forecast Auto-Detection Feature
auto_weather = st.sidebar.checkbox("🛰️ Auto-fetch Dehradun Forecast", value=True)
forecast_info = get_dehradun_forecast(selected_date, hour)

weather_icons = {
    'Clear': '☀️ Clear Skies',
    'Cloudy': '⛅ Overcast / Cloudy',
    'Fog / Haze': '🌫️ Morning Fog / Haze',
    'Rain': '🌧️ Rain / Wet Roads'
}
weather_keys = list(weather_icons.keys())

if auto_weather:
    weather = forecast_info['condition']
    st.sidebar.info(
        f"**Forecast for {hour:02d}:00**:\n\n"
        f"**{weather_icons.get(weather, weather)}**\n\n"
        f"🌡️ **Temp**: {forecast_info['temp_c']}°C &nbsp;|&nbsp; 🌧️ **Rain Chance**: {forecast_info['rain_prob']}%\n\n"
        f"_{forecast_info['source']}_",
        icon="📡"
    )
else:
    selected_weather_display = st.sidebar.selectbox(
        "Simulate Weather Scenario",
        [weather_icons[k] for k in weather_keys],
        index=0
    )
    weather = [k for k, v in weather_icons.items() if v == selected_weather_display][0]

st.sidebar.markdown("---")
st.sidebar.info("💡 **SmartRoute AIML Project**\nUPES Bidholi ↔ Clock Tower Dehradun & Corridors.")

# -------------------------------------------------------------
# MAIN APP BODY
# -------------------------------------------------------------
st.markdown('<div class="main-header">SmartRoute: Travel Time Prediction System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Accurate journey duration estimation using ML regression trained on real TomTom traffic observations.</div>', unsafe_allow_html=True)

# Tabs
tab_predict, tab_analytics, tab_models = st.tabs([
    "🚀 Journey Prediction",
    "📈 24-Hour Traffic Curve",
    "🏆 ML Model Benchmark & Report"
])

# Compute prediction
pred = predictor.predict_trip(
    origin=origin,
    destination=destination,
    hour=hour,
    minute=minute,
    day_of_week=selected_day,
    weather=weather
)

# Calculate Arrival Time
dep_dt = datetime.datetime.combine(selected_date, datetime.time(hour, minute))
arr_dt = dep_dt + datetime.timedelta(minutes=pred['predicted_duration_minutes'])
arr_str = arr_dt.strftime("%I:%M %p")
dep_str = dep_dt.strftime("%I:%M %p")

# =============================================================
# TAB 1: PREDICTION DASHBOARD
# =============================================================
with tab_predict:
    st.subheader(f"📍 Journey Summary: **{origin}** ➔ **{destination}**")
    st.caption(f"Departure on **{selected_date.strftime('%d %b %Y')} ({selected_day})** at **{dep_str}** | Weather: **{weather}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Estimated Duration",
            value=f"{pred['predicted_duration_minutes']} min",
            delta=f"Range: {pred['lower_bound_minutes']} - {pred['upper_bound_minutes']} min",
            delta_color="off"
        )
        
    with col2:
        st.metric(
            label="Expected Arrival",
            value=arr_str,
            delta=f"Leaves at {dep_str}",
            delta_color="normal"
        )
        
    with col3:
        st.metric(
            label="Route Distance",
            value=f"{pred['distance_km']} km",
            delta="TomTom Route Length"
        )
        
    with col4:
        st.metric(
            label="Average Speed",
            value=f"{pred['speed_kmh']} km/h",
            delta=pred['traffic_status']
        )
        
    st.markdown("---")
    
    # Traffic Level Alert Card
    status_class = "badge-green" if "Smooth" in pred['traffic_status'] else ("badge-orange" if "Moderate" in pred['traffic_status'] else "badge-red")
    st.markdown(f"""
    <div style="background-color: #F3F4F6; padding: 14px 18px; border-radius: 8px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <strong>Traffic Condition:</strong> <span class="status-badge {status_class}">{pred['traffic_status']}</span>
            &nbsp;|&nbsp; <strong>Peak Rush Period:</strong> {'⚠️ Yes (Expect Delays)' if pred['is_rush_hour'] else '✅ No (Off-Peak)'}
        </div>
        <div style="font-size: 0.9rem; color: #4B5563;">
            Weather: <strong>{weather}</strong> {f"({forecast_info['temp_c']}°C)" if auto_weather else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Route Waypoints Map
    col_map, col_details = st.columns([1.3, 1])
    
    with col_map:
        st.markdown("##### 🗺️ Route Corridor Map")
        map_df = pd.DataFrame([
            {'name': origin, 'lat': COORDINATES[origin]['lat'], 'lon': COORDINATES[origin]['lon']},
            {'name': destination, 'lat': COORDINATES[destination]['lat'], 'lon': COORDINATES[destination]['lon']}
        ])
        st.map(map_df, zoom=11)
        
    with col_details:
        st.markdown("##### 📋 Trip Analysis & Practical Advice")
        if pred['is_rush_hour']:
            st.warning("⚠️ **Rush Hour Alert**: You are traveling during peak commuter traffic. Premnagar and Chakrata Road bottlenecks may experience slow moving traffic.")
        else:
            st.success("✅ **Favorable Commute Time**: Road conditions are relatively clear with minimal signal congestion.")
            
        if weather == 'Rain':
            st.info("🌧️ **Wet Weather Notice**: TomTom historical trends show ~15% higher journey durations during rainfall due to hill road cautions.")
        elif weather == 'Fog / Haze':
            st.info("🌫️ **Fog Warning**: Visibility on the Bidholi hill climb may cause drivers to reduce average speeds.")
            
        st.markdown(f"""
        - **Origin Node**: {origin}
        - **Destination Node**: {destination}
        - **Model Confidence**: ±1.5 minutes (MAE: 0.92 min)
        """)

# =============================================================
# TAB 2: 24-HOUR TRAFFIC CURVE
# =============================================================
with tab_analytics:
    st.subheader(f"📊 24-Hour Traffic Pattern: {origin} ➔ {destination}")
    st.markdown(f"Shows the expected travel time for every hour of **{selected_day}** under **{weather}** conditions.")
    
    # Compute full day profile
    full_day_df = predictor.predict_full_day_profile(origin, destination, selected_day, weather)
    
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(full_day_df['fractional_hour'], full_day_df['duration_minutes'], color='#2563EB', lw=2.5, label='Predicted Duration')
    ax.fill_between(full_day_df['fractional_hour'], full_day_df['duration_minutes'] - 1.5, full_day_df['duration_minutes'] + 1.5, color='#93C5FD', alpha=0.3, label='Confidence Interval (±1.5m)')
    
    # Highlight user's chosen time
    user_time_frac = hour + minute / 60.0
    user_duration = pred['predicted_duration_minutes']
    ax.scatter([user_time_frac], [user_duration], color='#DC2626', s=120, zorder=5, label=f'Your Departure ({dep_str})')
    
    # Highlight rush hour zones
    ax.axvspan(8, 11, color='#FEF08A', alpha=0.25, label='Morning Peak (8-11 AM)')
    ax.axvspan(17, 20, color='#FED7AA', alpha=0.25, label='Evening Peak (5-8 PM)')
    
    ax.set_title(f'Travel Duration Throughout {selected_day}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=10)
    ax.set_ylabel('Duration (minutes)', fontsize=10)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', frameon=True)
    st.pyplot(fig)
    
    # Best Departure Time Recommendation
    fastest_time = full_day_df.loc[full_day_df['duration_minutes'].idxmin()]
    slowest_time = full_day_df.loc[full_day_df['duration_minutes'].idxmax()]
    
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.success(f"🟢 **Fastest Time to Travel**: **{fastest_time['time_label']}** (~{fastest_time['duration_minutes']} mins)")
    with col_rec2:
        st.error(f"🔴 **Peak Congestion Time**: **{slowest_time['time_label']}** (~{slowest_time['duration_minutes']} mins)")

# =============================================================
# TAB 3: ML BENCHMARK & PROJECT ARTIFACTS
# =============================================================
with tab_models:
    st.subheader("🏆 Model Benchmark & Evaluation Report")
    st.markdown("Comparative performance analysis of regression models evaluated on the held-out test set.")
    
    # Benchmark Table
    metrics_data = predictor.metadata['metrics']
    benchmark_df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={'index': 'Model Architecture'})
    
    st.dataframe(
        benchmark_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#D1FAE5')
                          .highlight_max(subset=['R2'], color='#D1FAE5'),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("##### 📈 Visual Artifacts for Final Presentation")
    
    col_fig1, col_fig2 = st.columns(2)
    fig_dir = os.path.join(os.path.dirname(__file__), 'reports', 'figures')
    
    model_comp_path = os.path.join(fig_dir, 'model_comparison.png')
    parity_path = os.path.join(fig_dir, 'actual_vs_predicted.png')
    feat_imp_path = os.path.join(fig_dir, 'feature_importance.png')
    hourly_prof_path = os.path.join(fig_dir, 'hourly_congestion_profile.png')
    
    if os.path.exists(model_comp_path):
        with col_fig1:
            st.image(model_comp_path, caption="Figure 1: Cross-Model MAE and R² Score Comparison")
            
    if os.path.exists(parity_path):
        with col_fig2:
            st.image(parity_path, caption="Figure 2: Actual vs. Predicted Travel Duration Parity Plot")
            
    col_fig3, col_fig4 = st.columns(2)
    if os.path.exists(feat_imp_path):
        with col_fig3:
            st.image(feat_imp_path, caption="Figure 3: Feature Importance Analysis")
            
    if os.path.exists(hourly_prof_path):
        with col_fig4:
            st.image(hourly_prof_path, caption="Figure 4: Corridor Congestion Hourly Profile")
            
    st.markdown("---")
    st.markdown("### 🎓 Semester Presentation Summary")
    st.markdown("""
    - **Dataset**: 7,984 real-world TomTom Orbis Routing API observations across Dehradun travel hubs.
    - **Champion Model**: Gradient Boosting Regressor ($R^2 = 0.9901$, MAE = 0.92 minutes).
    - **Key Findings**: Departure hour and distance dominate journey duration; morning and evening rush hour effects create up to 40% variance in travel times for the UPES Bidholi corridor.
    """)
