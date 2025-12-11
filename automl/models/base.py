from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseTimeSeriesModel(ABC):
    """Abstract base class for all time series models with unified interface"""

    def __init__(self, name, model_type, params=None):
        self.name = name
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, series, X=None, y=None):
        """Fit the model to the data"""
        pass

    @abstractmethod
    def predict(self, steps=None, X=None):
        """Generate predictions/forecasts"""
        pass

    @abstractmethod
    def forecast(self, series, steps):
        """Generate forecasts from a trained model"""
        pass

    def evaluate(self, y_true, y_pred, metric="rmse"):
        """Evaluate model predictions"""
        if metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "r2":
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_params(self):
        """Get model parameters"""
        return self.params

    def set_params(self, **params):
        """Set model parameters"""
        self.params.update(params)
        return self


class StatisticalModel(BaseTimeSeriesModel):
    """Wrapper for statistical models with unified interface"""

    def __init__(self, name, model_class, params=None):
        super().__init__(name, "statistical", params)
        self.model_class = model_class
        self.fitted_model = None

    def fit(self, series, X=None, y=None):
        """Fit statistical model to series"""
        if self.name == "ARIMA":
            order = self.params.get("order", (1, 1, 1))
            self.fitted_model = self.model_class(series, order=order).fit()
        elif self.name == "SARIMA":
            order = self.params.get("order", (1, 1, 1))
            seasonal_order = self.params.get("seasonal_order", (0, 0, 0, 0))
            self.fitted_model = self.model_class(
                series, order=order, seasonal_order=seasonal_order
            ).fit()
        elif self.name == "Exponential Smoothing":
            trend = self.params.get("trend", "add")
            seasonal = self.params.get("seasonal", "add")
            seasonal_periods = self.params.get("seasonal_periods", 12)
            self.fitted_model = self.model_class(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            ).fit()
        else:
            # Generic statistical model
            self.fitted_model = self.model_class(series, **self.params).fit()

        self.is_fitted = True
        return self

    def predict(self, steps=None, X=None):
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if steps:
            return self.fitted_model.forecast(steps=steps)
        else:
            return self.fitted_model.predict()

    def forecast(self, series, steps):
        """Generate forecasts from scratch"""
        self.fit(series)
        return self.predict(steps=steps)


class MLModel(BaseTimeSeriesModel):
    """Wrapper for machine learning models with unified interface"""

    def __init__(self, name, model_class, params=None):
        super().__init__(name, "ml", params)
        self.model_class = model_class
        self.model_instance = None

    def fit(self, series, X, y):
        """Fit ML model to features"""
        self.model_instance = self.model_class(**self.params)
        self.model_instance.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, steps=None, X=None):
        """Generate predictions on features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X is not None:
            return self.model_instance.predict(X)
        else:
            raise ValueError("X must be provided for ML model predictions")

    def forecast(self, series, steps):
        """ML models need features for forecasting - this is handled elsewhere"""
        raise NotImplementedError(
            "ML models require feature engineering for forecasting. "
            "Use predict() with feature matrix X instead."
        )


class ModelResult:
    """Container for model training results"""

    def __init__(
        self, model_name, model_type, model_instance, params, score, features_shape=None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.model_instance = model_instance
        self.params = params
        self.score = score
        self.features_shape = features_shape

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model": self.model_instance,
            "params": self.params,
            "score": self.score,
            "features_shape": self.features_shape,
        }
