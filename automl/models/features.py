import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from statsmodels.tsa.statespace.sarimax import SARIMAX


class FeatureEngineering:
    """Advanced feature engineering for time series with configurable options"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names = []

    def create_features(
        self,
        series: pd.DataFrame,
        feature_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create all configured features"""
        dates = series.index
        if feature_types is None:
            feature_types = ["lag", "seasonal", "fourier"]
            #feature_types = ["lag", "seasonal", "fourier", "rolling"]

        feature_df = series.copy()

        features_list = [self.create_timestamps(feature_df, dates)]

        # Lag features
        if "lag" in feature_types:
            lag_config = self.config.get("lag_features", {})
            max_lags = lag_config.get("max_lags", 3)
            lag_features = self.create_lag_features(feature_df, series, max_lags)
            features_list.append(lag_features)

        # Substitute features using (S)ARIMA
        if "substitute" in feature_types:
            substitute_config = self.config.get("substitute_features", {})
            substitute_features = self.create_substitute_features(
                feature_df, series, substitute_config
            )
            features_list.append(substitute_features)

        # Rolling features
        if "rolling" in feature_types:
            rolling_config = self.config.get("rolling_features", {})
            windows = rolling_config.get("windows", [30, 60, 90])
            stats = rolling_config.get("stats", ["mean", "std", "min", "max"])
            rolling_features = self.create_rolling_features(
                feature_df, series, windows, stats
            )
            features_list.append(rolling_features)

        feature_df = feature_df.fillna(0)

        # Seasonal features
        if "seasonal" in feature_types:
            seasonal_features = self.create_seasonal_features(feature_df, dates)
            features_list.append(seasonal_features)

            # Fourier features
            if self.config.get("seasonal_features", {}).get("include_fourier", True):
                fourier_config = self.config.get("seasonal_features", {})
                periods = fourier_config.get("fourier_periods", [7, 365])
                n_terms = fourier_config.get("fourier_terms", 2)

                for period in periods:
                    if len(series) > period:
                        fourier_features = self.create_fourier_features(
                            feature_df, dates, period, n_terms
                        )
                        features_list.append(fourier_features)

        # delete answer(s)
        for target in series.columns:
            del feature_df[target]

        return feature_df

    def create_substitute_features(
        self, feature_df, series: pd.DataFrame, substitute_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create substitute features using (S)ARIMA to predict target values
        using only previous observations.
        """
        if substitute_config is None:
            substitute_config = self.config.get("substitute_features", {})

        # Configuration with defaults
        cold_start = substitute_config.get(
            "cold_start", 100
        )  # First n points use raw lags
        k_windows = substitute_config.get("k_windows", 3)  # Number of fitting windows
        min_fit_size = substitute_config.get(
            "min_fit_size", 50
        )  # Minimum data to fit model
        order = substitute_config.get("order", (1, 0, 1))  # ARIMA order
        seasonal_order = substitute_config.get(
            "seasonal_order", (1, 0, 1, 7)
        )  # SARIMA order

        substitute_df = pd.DataFrame(index=series.index)

        for target_col in series.columns:
            substitute_name = f"substitute_{target_col}"
            target_series = series[target_col]

            # Initialize substitute series with NaNs
            substitute_series = pd.Series(np.nan, index=target_series.index)

            # Separate data into fitting region and prediction region
            # Avoid using tail zeros for fitting
            non_zero_mask = target_series != 0
            if non_zero_mask.any():
                last_non_zero_idx = target_series[non_zero_mask].index[-1]
                fitting_data = target_series.loc[:last_non_zero_idx]
            else:
                fitting_data = target_series

            for i in range(len(target_series)):
                current_idx = target_series.index[i]

                # Skip if we're in the tail zeros region (prediction mode)
                if (
                    i > 0
                    and target_series.iloc[i] == 0
                    and target_series.iloc[i - 1] == 0
                    and not target_series[i:].any()
                ):
                    steps = len(target_series) - i + 1
                    # Check if we're in continuous zero tail
                    # Final SARIMA model fit
                    model = SARIMAX(
                        train_data,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=True,  # More efficient
                    )
                    with np.errstate(all="ignore"):
                        fit_result = model.fit(
                            disp=False, maxiter=50, method="lbfgs", start_params=None
                        )

                    # Forecast next value
                    forecast = fit_result.get_forecast(steps=steps)
                    for j in range(i, len(target_series)):
                        substitute_series.iloc[j] = forecast.predicted_mean.iloc[j - i]
                    break

                # Cold start: use raw lags for first n points
                if i < cold_start:
                    if i > 0:
                        substitute_series.iloc[i] = target_series.iloc[i - 1]
                    continue

                # For computational efficiency, only fit models at specific intervals
                # One prediction window needs at least two fit windows
                fit_interval = max(1, len(fitting_data) // (k_windows * 2))
                if i % fit_interval != 0 and i > cold_start:
                    # Use last fitted model's prediction or extrapolate
                    if not pd.isna(substitute_series.iloc[i - 1]):
                        substitute_series.iloc[i] = substitute_series.iloc[i - 1]
                    else:
                        substitute_series.iloc[i] = target_series.iloc[i - 1]
                    continue

                # Check if we have enough data to fit the model
                if (
                    i < min_fit_size * 2
                ):  # Need at least 2*min_fit_size for proper fitting
                    substitute_series.iloc[i] = target_series.iloc[i - 1]
                    continue

                try:
                    # Use data up to current point for fitting
                    train_data = target_series.iloc[:i]

                    # Check if data has enough variation to fit model
                    if train_data.std() < 1e-10 or len(train_data) < min_fit_size:
                        substitute_series.iloc[i] = target_series.iloc[i - 1]
                        continue

                    # Fit SARIMA model
                    model = SARIMAX(
                        train_data,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=True,  # More efficient
                    )

                    # Use minimal fitting to save computation
                    with np.errstate(all="ignore"):
                        fit_result = model.fit(
                            disp=False, maxiter=50, method="lbfgs", start_params=None
                        )

                    # Forecast next value
                    forecast = fit_result.get_forecast(steps=1)
                    predicted_value = forecast.predicted_mean.iloc[0]

                    # Ensure prediction is reasonable
                    if np.isfinite(predicted_value):
                        # Blend with last actual value to reduce shock
                        #blend_ratio = 0.7
                        blend_ratio = 1
                        substitute_series.iloc[i] = (
                            blend_ratio * predicted_value
                            + (1 - blend_ratio) * target_series.iloc[i - 1]
                        )
                    else:
                        substitute_series.iloc[i] = target_series.iloc[i - 1]

                except Exception as e:
                    # Fall back to raw lag if model fails
                    substitute_series.iloc[i] = target_series.iloc[i - 1]

            # Forward fill any remaining NaNs
            substitute_series = substitute_series.fillna(0)

            substitute_df[substitute_name] = substitute_series
            self.feature_names.append(substitute_name)

        for column in substitute_df.columns:
            feature_df[column] = substitute_df[column]

        return substitute_df

    @staticmethod
    def create_lag_features(df, series: pd.Series, max_lags: int = 3) -> pd.DataFrame:
        """Create lag features for time series"""
        RATE = 30
        for lag in range(RATE, RATE * max_lags + 1, RATE):
            shifted = series.shift(lag)
            for column in series.shift(lag):
                df[f"lag_{lag}_{column}"] = shifted[column]
        return df

    @staticmethod
    def create_rolling_features(
        df, series: pd.Series, windows: List[int] = None, stats: List[str] = None
    ) -> pd.DataFrame:
        """Create rolling statistics features"""
        RATE = 30
        if windows is None:
            windows = [10, 15, 30]
        if stats is None:
            stats = ["mean", "std", "min", "max"]
        shifted_series = series.shift(RATE)

        stat_functions = {
            "mean": lambda w: shifted_series.rolling(window=w).mean(),
            "std": lambda w: shifted_series.rolling(window=w).std(),
            "min": lambda w: shifted_series.rolling(window=w).min(),
            "max": lambda w: shifted_series.rolling(window=w).max(),
            "median": lambda w: shifted_series.rolling(window=w).median(),
            "sum": lambda w: shifted_series.rolling(window=w).sum(),
        }

        for window in windows:
            for stat in stats:
                if stat in stat_functions:
                    df[f"rolling_{stat}_{window}"] = stat_functions[stat](window)
        df.fillna(0)

        return df

    @staticmethod
    def create_seasonal_features(df, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create date-based seasonal features"""
        df["hour"] = dates.hour
        df["day"] = dates.day
        df["dayofweek"] = dates.dayofweek
        df["dayofyear"] = dates.dayofyear
        df["week"] = dates.isocalendar().week.astype(int)
        df["month"] = dates.month
        df["quarter"] = dates.quarter
        df["year"] = dates.year
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        return df

    @staticmethod
    def create_timestamps(df, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Add simple timestamp"""
        df["timestamp"] = pd.Index(map(lambda d: d.timestamp(), dates), name=dates.name)

        return df

    @staticmethod
    def create_fourier_features(
        df, dates: pd.DatetimeIndex, period: int, n_terms: int = 3
    ) -> pd.DataFrame:
        """Create Fourier terms for seasonality"""
        t = np.arange(len(dates))
        for i in range(1, n_terms + 1):
            df[f"fourier_sin_{period}_{i}"] = np.sin(2 * np.pi * i * t / period)
            df[f"fourier_cos_{period}_{i}"] = np.cos(2 * np.pi * i * t / period)
        return df

    def create_future_features(
        self,
        future_dates: pd.DatetimeIndex,
        series: pd.DataFrame = None,
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """Create features for future dates (for forecasting)"""
        size = len(future_dates)
        data = {col: [0] * size for col in series}
        dummy_series = pd.DataFrame(data, index=future_dates)

        combined = pd.concat([series, dummy_series])
        combined.index.freq = future_dates.freq
        df = self.create_features(
            combined,
        )

        # Select only the features we have names for
        if feature_names is not None:
            for col in df.columns:
                if col not in feature_names:
                    del df[col]
            return df
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names"""
        return self.feature_names.copy()
