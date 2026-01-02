import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from automl.models.features import FeatureEngineering

dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
series = pd.Series(np.sin(np.arange(60)/5) + np.random.normal(0, 0.1, 60), index=dates, name="y")

feature_engineer = FeatureEngineering(config={
    "lag_features": {"max_lags": 5},
    "rolling_features": {"windows": [3], "stats": ["mean"]},
    "seasonal_features": {"include_fourier": False}
})

X_all = feature_engineer.create_features(series, dates)
y_all = series.loc[X_all.index]

train_size = 50
X_train, y_train = X_all.iloc[:train_size], y_all.iloc[:train_size]
X_test, y_test = X_all.iloc[train_size:], y_all.iloc[train_size:]

ml_models = {
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
}

stat_models = {
    "ARIMA": ARIMA(series.iloc[:train_size], order=(2,1,2)),
    "Exponential Smoothing": ExponentialSmoothing(series.iloc[:train_size], seasonal="add", seasonal_periods=7),
}

forecast_steps = 7
results = {}

# --- Forecast iterativo para ML ---
for name, model in ml_models.items():
    series_copy = series.iloc[:train_size].copy()
    preds = []
    model.fit(X_train, y_train)
    
    for step in range(forecast_steps):
        future_date = pd.date_range(series_copy.index[-1] + pd.Timedelta(days=1), periods=1)
        X_future = feature_engineer.create_future_features(
            index=None,
            future_dates=future_date,
            series=series_copy,
            feature_names=feature_engineer.get_feature_names()
        )
        pred = model.predict(X_future)[0]
        preds.append(pred)
        series_copy = pd.concat([series_copy, pd.Series([pred], index=future_date)])
    
    forecast_index = pd.date_range(series.index[train_size], periods=forecast_steps)
    results[name] = pd.Series(preds, index=forecast_index)

# ---Forecast para modelos estadísticos ---
for name, model in stat_models.items():
    fit_model = model.fit()
    forecast_index = pd.date_range(series.index[train_size], periods=forecast_steps)
    forecast = fit_model.forecast(steps=forecast_steps)
    results[name] = pd.Series(forecast.values, index=forecast_index)

# --- Mostrar resultados ---
for name, forecast in results.items():
    print(f"\n{name} forecast:")
    print(forecast)

# --- Gráfico comparativo ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(series, label="Original Series", marker='o')
for name, forecast in results.items():
    plt.plot(forecast, label=f"{name} Forecast", marker='x')
plt.axvline(series.index[train_size], color='gray', linestyle='--', label="Forecast Start")
plt.legend()
plt.show()
