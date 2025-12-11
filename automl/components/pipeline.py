import streamlit as st
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.base import BaseTimeSeriesModel, ModelResult
from models.factory import ModelFactory
from models.features import FeatureEngineering


class AutoMLPipeline:
    """Unified AutoML engine for time series forecasting"""

    def __init__(
        self,
        model_registry: Dict[str, Any],
        feature_config: Optional[Dict[str, Any]] = None,
        strategy: str = "optuna_bayesian",
        metric: str = "rmse",
        freq: str = "D",
        cv_splits: int = 5,
    ):

        self.model_registry = model_registry
        self.feature_engineer = FeatureEngineering(feature_config)
        self.strategy = strategy
        self.metric = metric
        self.freq = freq
        self.cv_splits = cv_splits
        self.best_models: List[ModelResult] = []
        self.X_features: Optional[pd.DataFrame] = None
        self.y_target: Optional[pd.Series] = None

    def prepare_features(
        self,
        series: pd.Series,
        dates: pd.DatetimeIndex,
        feature_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features based on configuration"""
        if feature_options is None:
            feature_options = {
                "lag_features": False,
                "rolling_features": False,
                "seasonal_features": True,
                "max_lags": 12,
            }

        # Determine which feature types to create
        feature_types = []
        if feature_options.get("lag_features", True):
            feature_types.append("lag")
            self.feature_engineer.config["lag_features"] = {
                "max_lags": feature_options.get("max_lags", 12)
            }

        if feature_options.get("rolling_features", True):
            feature_types.append("rolling")

        if feature_options.get("seasonal_features", True):
            feature_types.append("seasonal")
            feature_types.append("fourier")

        # Create features
        X = self.feature_engineer.create_features(series, dates, feature_types)

        if X.empty:
            return X, series

        # Align target with features
        y = series.loc[X.index]

        self.X_features = X
        self.y_target = y

        return X, y

    def optimize_model(
        self,
        model_name: str,
        X: Optional[pd.DataFrame],
        y: pd.Series,
        n_trials: int = 50,
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize a model using Optuna"""

        def objective(trial):
            params = {}
            model_config = self.model_registry[model_name]

            # Sample parameters
            for param_name, param_values in model_config["params"].items():
                if isinstance(param_values[0], (str, bool, type(None))):
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
                elif isinstance(param_values[0], tuple):
                    if param_name == "hidden_layer_sizes":
                        n_layers = trial.suggest_int("n_layers", 1, 3)
                        layer_sizes = []
                        for i in range(n_layers):
                            size = trial.suggest_int(f"layer_{i}_size", 10, 100)
                            layer_sizes.append(size)
                        params[param_name] = tuple(layer_sizes)
                    else:
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_values
                        )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )

            try:
                model = ModelFactory.create_model(
                    {**model_config, "name": model_name, "params": params}
                )

                if model.model_type == "statistical":
                    # Evaluate statistical models
                    return self._evaluate_statistical_model(model, y)
                else:
                    # Evaluate ML models with cross-validation
                    tscv = TimeSeriesSplit(n_splits=self.cv_splits)

                    # Create temporary sklearn-compatible model for CV
                    from sklearn.base import BaseEstimator, RegressorMixin

                    class SklearnWrapper(BaseEstimator, RegressorMixin):
                        def __init__(self, **kwargs):
                            self.params = kwargs
                            self.model = ModelFactory.create_model(
                                {**model_config, "name": model_name, "params": kwargs}
                            )
                            self.is_fitted = False

                        def fit(self, X, y):
                            self.model.fit(pd.Series(y), X, y)
                            self.is_fitted = True
                            return self

                        def predict(self, X):
                            return self.model.predict(X=X)

                    wrapped_model = SklearnWrapper(**params)

                    if self.metric == "rmse":
                        scores = cross_val_score(
                            wrapped_model,
                            X,
                            y,
                            cv=tscv,
                            scoring="neg_root_mean_squared_error",
                        )
                        return -scores.mean()
                    elif self.metric == "mae":
                        scores = cross_val_score(
                            wrapped_model,
                            X,
                            y,
                            cv=tscv,
                            scoring="neg_mean_absolute_error",
                        )
                        return -scores.mean()
                    else:  # r2
                        scores = cross_val_score(
                            wrapped_model, X, y, cv=tscv, scoring="r2"
                        )
                        return -scores.mean()

            except Exception as e:
                return float("inf")

        # Run optimization
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params, study.best_value

    def _evaluate_statistical_model(
        self, model: BaseTimeSeriesModel, series: pd.Series
    ) -> float:
        """Evaluate statistical model"""
        try:
            train_size = int(len(series) * 0.8)
            train, test = series.iloc[:train_size], series.iloc[train_size:]

            model.fit(train)
            forecast = model.predict(steps=len(test))

            if self.metric == "rmse":
                return np.sqrt(mean_squared_error(test, forecast))
            elif self.metric == "mae":
                return mean_absolute_error(test, forecast)
            else:  # r2
                return 1 - r2_score(test, forecast)

        except Exception as e:
            return float("inf")

    def run(
        self,
        series: pd.Series,
        dates: pd.DatetimeIndex,
        selected_models: List[str],
        feature_options: Optional[Dict[str, Any]] = None,
        n_trials_per_model: int = 30,
    ) -> List[ModelResult]:
        """Run the complete AutoML pipeline"""

        results = []

        # Prepare features for ML models
        X, y = self.prepare_features(series, dates, feature_options)

        for model_name in selected_models:
            if model_name not in self.model_registry:
                st.warning(f"Model '{model_name}' not found in registry. Skipping.")
                continue

            with st.spinner(f"Optimizing {model_name}..."):
                model_config = self.model_registry[model_name]

                if model_config["type"] == "statistical":
                    # Optimize statistical model
                    best_params, best_score = self.optimize_model(
                        model_name, None, series, n_trials=n_trials_per_model
                    )

                    # Create and fit final model
                    model = ModelFactory.create_model(
                        {**model_config, "name": model_name, "params": best_params}
                    )
                    model.fit(series)

                    results.append(
                        ModelResult(
                            model_name=model_name,
                            model_type="statistical",
                            model_instance=model,
                            params=best_params,
                            score=best_score,
                        )
                    )

                else:  # ML model
                    if X is not None and not X.empty:
                        best_params, best_score = self.optimize_model(
                            model_name, X, y, n_trials=n_trials_per_model
                        )

                        # Create and fit final model
                        model = ModelFactory.create_model(
                            {**model_config, "name": model_name, "params": best_params}
                        )
                        model.fit(series, X, y)

                        results.append(
                            ModelResult(
                                model_name=model_name,
                                model_type="ml",
                                model_instance=model,
                                params=best_params,
                                score=best_score,
                                features_shape=X.shape,
                            )
                        )
                    else:
                        st.warning(f"Cannot train {model_name}: No features generated")

        # Sort results by score
        results.sort(key=lambda x: x.score)
        self.best_models = results

        return results

    def forecast(
        self,
        model_result: ModelResult,
        series: pd.Series,
        dates: pd.DatetimeIndex,
        steps: int,
    ) -> pd.Series:
        """Generate forecasts using a trained model"""
        model = model_result.model_instance

        if model_result.model_type == "statistical":
            # Statistical models forecast directly
            return model.predict(steps=steps)
        else:
            # ML models need future features
            last_date = dates[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=min(steps, len(series) // 2),
                freq=dates.freq or self.freq,
            )

            # Create future features
            future_features = self.feature_engineer.create_future_features(
                model.model_instance.feature_names_in_,
                future_dates,
                series=series,
            )
            if future_features.empty:
                raise ValueError("Could not create features for forecasting")

            # Predict using ML model
            predictions = model.predict(X=future_features)

            return pd.Series(predictions, index=future_dates, name=series.name)

    def add_custom_model(self, name: str, model_config: Dict[str, Any]) -> None:
        """Add a custom model to the registry"""
        ModelFactory.validate_model_config(model_config)
        self.model_registry[name] = model_config

    def remove_model(self, name: str) -> None:
        """Remove a model from the registry"""
        if name in self.model_registry:
            del self.model_registry[name]

    def get_model_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.model_registry.get(name)

    def update_model_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific model"""
        if name in self.model_registry:
            self.model_registry[name].update(config)
        else:
            raise ValueError(f"Model '{name}' not found in registry")
