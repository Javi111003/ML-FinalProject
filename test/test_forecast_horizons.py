import pandas as pd
import numpy as np
from automl.components.pipeline import AutoMLPipeline
from automl.models.factory import ModelFactory

# =========================
# 1. Crear serie sintética
# =========================
dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
np.random.seed(42)

series = pd.Series(
    0.1 * np.arange(60) + np.sin(np.arange(60) / 5) + np.random.normal(0, 0.1, 60),
    index=dates,
    name="value"
)

# =========================
# 2. Configurar modelos
# =========================
model_registry = {
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "type": "ml",
        "params": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    },
    "Gradient Boosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "type": "ml",
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    },
    "ARIMA": {
        "class": "statsmodels.tsa.arima.model.ARIMA",
        "type": "statistical",
        "params": {"order": [(1,1,1), (2,1,2)]},
    },
    "Exponential Smoothing": {
        "class": "statsmodels.tsa.holtwinters.ExponentialSmoothing",
        "type": "statistical",
        "params": {"trend": ["add", "mul"], "seasonal": ["add", "mul"], "seasonal_periods": [7]},
    },
}

# =========================
# 3. Configurar pipeline
# =========================
pipeline = AutoMLPipeline(model_registry=model_registry, freq="D", metric="rmse")

# =========================
# 4. Configurar feature engineering
# =========================
feature_options = {
    "lag_features": True,
    "max_lags": 5,
    "rolling_features": True,
    "seasonal_features": True
}

# =========================
# 5. Ejecutar AutoML para todos los modelos
# =========================
selected_models = list(model_registry.keys())
best_models = pipeline.run(
    series=series,
    dates=dates,
    selected_models=selected_models,
    feature_options=feature_options,
    n_trials_per_model=3  # poco para test rápido
)

# =========================
# 6. Forecast de 7 días
# =========================
forecast_steps = 7
for model_result in best_models:
    forecast = pipeline.forecast(model_result, series, dates, forecast_steps)
    print(f"\n{model_result.model_name} forecast:")
    print(forecast)
