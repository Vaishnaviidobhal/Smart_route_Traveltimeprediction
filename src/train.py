"""
SmartRoute Model Training & Evaluation Pipeline
Trains multiple regression models (Linear Regression, Ridge, Decision Tree,
Random Forest, Gradient Boosting), benchmarks metrics (MAE, RMSE, R2, MAPE),
saves the top-performing pipeline, and produces publication-ready visual artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Configure high-quality aesthetic plots for presentations
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_traffic_data.csv')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"[INFO] Reading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Define features and target
    # Predicting 'duration_minutes'
    target_col = 'duration_minutes'
    
    categorical_features = ['origin', 'destination', 'day_of_week', 'weather']
    numeric_features = ['distance_km', 'hour', 'minute', 'fractional_hour', 'is_weekend', 'is_rush_hour']
    
    X = df[categorical_features + numeric_features]
    y = df[target_col]
    
    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print(f"[INFO] Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    
    # Preprocessor definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='drop'
    )
    
    # Candidate models to evaluate
    candidate_models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_split=10, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=6, random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingRegressor(max_iter=150, max_depth=10, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None
    best_y_pred = None
    
    print("\n" + "="*70)
    print(f"{'MODEL EVALUATION BENCHMARK':^70}")
    print("="*70)
    print(f"{'Model':<30} | {'MAE (min)':<10} | {'RMSE (min)':<11} | {'R2 Score':<9} | {'MAPE (%)':<8}")
    print("-"*70)
    
    for name, regressor in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        
        # Predict on Test Set
        y_pred = pipe.predict(X_test)
        
        # Compute Evaluation Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        results[name] = {
            'MAE': round(float(mae), 3),
            'RMSE': round(float(rmse), 3),
            'R2': round(float(r2), 4),
            'MAPE': round(float(mape), 2)
        }
        
        print(f"{name:<30} | {mae:<10.3f} | {rmse:<11.3f} | {r2:<9.4f} | {mape:<8.2f}%")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipe
            best_y_pred = y_pred
            
    print("="*70)
    print(f"[WINNER] Best Performing Model: {best_model_name} (R2 = {best_r2:.4f})")
    
    # Save the Champion Model Pipeline
    model_save_path = os.path.join(models_dir, 'smartroute_best_model.pkl')
    joblib.dump(best_pipeline, model_save_path)
    print(f"[SAVED] Champion Model Pipeline saved to: {model_save_path}")
    
    # Also save metadata & route reference info
    metadata = {
        'best_model': best_model_name,
        'metrics': results,
        'features': {
            'categorical': categorical_features,
            'numeric': numeric_features
        },
        'locations': sorted(list(set(df['origin'].unique().tolist() + df['destination'].unique().tolist()))),
        'routes': sorted(df['route_name'].unique().tolist()),
        'weather_types': ['Clear', 'Cloudy', 'Fog / Haze', 'Rain'],
        'distance_lookup': df.groupby(['origin', 'destination'])['distance_km'].mean().to_dict()
    }
    
    # Convert tuple keys to string keys for JSON serialization
    metadata['distance_lookup'] = {f"{k[0]} -> {k[1]}": round(float(v), 2) for k, v in metadata['distance_lookup'].items()}
    
    metadata_save_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Metadata and metrics saved to: {metadata_save_path}")
    
    # -------------------------------------------------------------
    # GENERATE PRESENTATION-READY VISUAL ARTIFACTS
    # -------------------------------------------------------------
    print("\n[INFO] Generating presentation charts in reports/figures/...")
    
    # 1. Model Benchmark Comparison Chart
    df_metrics = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: MAE (Lower is better)
    sns.barplot(data=df_metrics, x='Model', y='MAE', ax=axes[0], palette='Blues_r')
    axes[0].set_title('Mean Absolute Error (Minutes) - Lower is Better', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE (min)')
    axes[0].tick_params(axis='x', rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}m", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)
        
    # Subplot 2: R2 Score (Higher is better)
    sns.barplot(data=df_metrics, x='Model', y='R2', ax=axes[1], palette='Greens_r')
    axes[1].set_title('R² Score (Variance Explained) - Higher is Better', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=30)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)
        
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted Parity Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_y_pred, alpha=0.35, color='#2563EB', edgecolors='none', s=25)
    min_val = min(y_test.min(), best_y_pred.min())
    max_val = max(y_test.max(), best_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line (y = x)')
    plt.title(f'Actual vs. Predicted Travel Duration\n({best_model_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Actual Travel Duration (minutes)', fontsize=11)
    plt.ylabel('Predicted Travel Duration (minutes)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 3. Residual Distribution Plot
    residuals = y_test - best_y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='#10B981', bins=40)
    plt.axvline(0, color='red', linestyle='--', label='Zero Error Reference')
    plt.title('Residual (Prediction Error) Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Prediction Error (Actual - Predicted) in Minutes', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()
    
    # 4. Hourly Congestion Profile (UPES Bidholi <-> Clock Tower)
    plt.figure(figsize=(10, 5))
    target_routes = ['UPES Bidholi -> Clock Tower', 'Clock Tower -> UPES Bidholi']
    route_df = df[df['route_name'].isin(target_routes)]
    sns.lineplot(
        data=route_df, x='hour', y='duration_minutes', hue='route_name',
        marker='o', markersize=6, palette=['#1E40AF', '#DC2626'], linewidth=2.5
    )
    plt.title('Hourly Travel Time Pattern for Primary College Corridor\n(UPES Bidholi ↔ Clock Tower)', fontsize=13, fontweight='bold')
    plt.xlabel('Hour of Day (0 to 23)', fontsize=11)
    plt.ylabel('Average Travel Time (minutes)', fontsize=11)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hourly_congestion_profile.png'), dpi=300)
    plt.close()
    
    # 5. Feature Importance for Random Forest / Tree Models
    rf_model = candidate_models['Random Forest Regressor']
    # Extract encoded feature names
    encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = encoded_cat_names + numeric_features
    
    rf_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])
    rf_pipe.fit(X_train, y_train)
    importances = rf_pipe.named_steps['regressor'].feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(9, 5))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 10 Most Influential Features (Random Forest)', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print("[SUCCESS] All figures generated in reports/figures/ successfully!")
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
