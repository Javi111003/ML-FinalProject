import warnings
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import importlib

warnings.filterwarnings("ignore")


class AutoMLTimeSeries:
    """AutoML engine for time series forecasting"""

    def __init__(
        self, model_configs, strategy="grid_search", metric="rmse", cv_splits=5
    ):
        self.strategy = strategy
        self.metric = metric
        self.cv_splits = cv_splits
        self.best_models = {}
        self.model_configs = model_configs

    def _get_model_class(self, class_path):
        """Dynamically import model class from string path"""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def optimize_model(self, model_name, X, y, n_trials=5):
        """Optimize a single model using Optuna"""
        model_config = self.model_configs[model_name]
        model_class = self._get_model_class(model_config["class"])

        def objective(trial):
            params = {}

            # Sample parameters based on model type
            for param_name, param_values in model_config["params"].items():
                # bool is also int
                if isinstance(param_values[0], (str, bool)):
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
                elif isinstance(param_values[0], (int, np.integer)):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values), log=True
                    )
                elif isinstance(param_values[0], tuple) and len(param_values[0]) == 4:
                    # SARIMA seasonal order
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
                elif isinstance(param_values[0], tuple):
                    # Handle hidden layer sizes for neural networks
                    if param_name == "hidden_layer_sizes":
                        n_layers = trial.suggest_int("n_layers", 1, 3)
                        layer_sizes = []
                        for i in range(n_layers):
                            size = trial.suggest_int(f"layer_{i}_size", 10, 100)
                            layer_sizes.append(size)
                        params[param_name] = tuple(layer_sizes)

            try:
                # Handle statistical models differently
                if model_config["type"] == "statistical":
                    return self.evaluate_statistical_model(model_class, params, y)
                else:
                    # For ML models, use cross-validation
                    model = model_class(**params)
                    tscv = TimeSeriesSplit(n_splits=self.cv_splits)

                    if self.metric == "rmse":
                        scores = cross_val_score(
                            model, X, y, cv=tscv, scoring="neg_root_mean_squared_error"
                        )
                        return -scores.mean()
                    elif self.metric == "mae":
                        scores = cross_val_score(
                            model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
                        )
                        return -scores.mean()
                    else:  # r2
                        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
                        return -scores.mean()  # Negative because we minimize
            except Exception as e:
                return float("inf")

        # Run optimization
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params, study.best_value

    def evaluate_statistical_model(self, model_class, params, series):
        """Evaluate statistical models (ARIMA, ETS)"""
        try:
            train_size = int(len(series) * 0.8)
            train, test = series.iloc[:train_size], series.iloc[train_size:]

            if model_class.__name__ == "ARIMA":
                model = model_class(
                    train,
                    order=(params.get("p", 1), params.get("d", 1), params.get("q", 1)),
                )
            elif model_class.__name__ == "SARIMAX":
                model = model_class(
                    train,
                    order=params.get("order", (1, 1, 1)),
                    seasonal_order=params.get("seasonal_order", (0, 0, 0, 0)),
                )
            elif model_class.__name__ == "ExponentialSmoothing":
                model = model_class(
                    train,
                    trend=params.get("trend", "add"),
                    seasonal=params.get("seasonal", "add"),
                    seasonal_periods=params.get("seasonal_periods", 12),
                )
            else:
                return float("inf")

            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))

            if self.metric == "rmse":
                return np.sqrt(mean_squared_error(test, forecast))
            elif self.metric == "mae":
                return mean_absolute_error(test, forecast)
            else:
                return 1 - r2_score(test, forecast)

        except Exception as e:
            return float("inf")
