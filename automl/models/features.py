import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class FeatureEngineering:
    """Advanced feature engineering for time series with configurable options"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names = []

    def create_features(
        self,
        series: pd.Series,
        dates: pd.DatetimeIndex,
        feature_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create all configured features"""
        if feature_types is None:
            feature_types = ["lag", "rolling", "seasonal", "fourier"]

        features_list = [self.create_timestamps(dates)]

        # Lag features
        if "lag" in feature_types:
            lag_config = self.config.get("lag_features", {})
            max_lags = lag_config.get("max_lags", 12)
            lag_features = self.create_lag_features(series, max_lags)
            features_list.append(lag_features)

        # Rolling features
        if "rolling" in feature_types:
            rolling_config = self.config.get("rolling_features", {})
            windows = rolling_config.get("windows", [7, 14, 30])
            stats = rolling_config.get("stats", ["mean", "std", "min", "max"])
            rolling_features = self.create_rolling_features(series, windows, stats)
            features_list.append(rolling_features)

        # Seasonal features
        if "seasonal" in feature_types:
            seasonal_features = self.create_seasonal_features(dates)
            features_list.append(seasonal_features)

            # Fourier features
            if self.config.get("seasonal_features", {}).get("include_fourier", True):
                fourier_config = self.config.get("seasonal_features", {})
                periods = fourier_config.get("fourier_periods", [7, 365])
                n_terms = fourier_config.get("fourier_terms", 2)

                for period in periods:
                    if len(series) > period:
                        fourier_features = self.create_fourier_features(
                            dates, period, n_terms
                        )
                        features_list.append(fourier_features)
        # Combine all features
        if features_list:
            features_df = pd.concat(features_list, axis=1)
            features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
            # delete answer!
            if series.name in features_df:
                del features_df[series.name]
            # features_df.index.name = series.index.name
            self.feature_names = list(features_df.columns)
            return features_df

        return pd.DataFrame(index=series.index)

    @staticmethod
    def create_lag_features(series: pd.Series, max_lags: int = 24) -> pd.DataFrame:
        """Create lag features for time series"""
        df = pd.DataFrame(series)
        for lag in range(1, max_lags + 1):
            # XXX fill_value
            df[f"lag_{lag}"] = series.shift(lag, fill_value=series.median())
        df = df.fillna(series.median())
        return df

    @staticmethod
    def create_rolling_features(
        series: pd.Series, windows: List[int] = None, stats: List[str] = None
    ) -> pd.DataFrame:
        """Create rolling statistics features"""
        if windows is None:
            windows = [7, 14, 30]
        if stats is None:
            stats = ["mean", "std", "min", "max"]

        df = pd.DataFrame(series)

        stat_functions = {
            "mean": lambda window: pd.Series.rolling(series, window=window).mean,
            "std": lambda window: pd.Series.rolling(series, window=window).std,
            "min": lambda window: pd.Series.rolling(series, window=window).min,
            "max": lambda window: pd.Series.rolling(series, window=window).max,
            "median": lambda window: pd.Series.rolling(series, window=window).median,
            "sum": lambda window: pd.Series.rolling(series, window=window).sum,
        }

        for window in windows:
            for stat in stats:
                if stat in stat_functions:
                    df[f"rolling_{stat}_{window}"] = stat_functions[stat](window)()
        # XXX fillna
        df = df.fillna(series.median())

        return df

    @staticmethod
    def create_seasonal_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create date-based seasonal features"""
        df = pd.DataFrame(index=dates)
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
    def create_timestamps(dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Add simple timestamp"""
        df = pd.DataFrame(index=dates)
        df["timestamp"] = pd.Index(map(lambda d: d.timestamp(), dates), name=dates.name)

        return df

    @staticmethod
    def create_fourier_features(
        dates: pd.DatetimeIndex, period: int, n_terms: int = 3
    ) -> pd.DataFrame:
        """Create Fourier terms for seasonality"""
        df = pd.DataFrame(index=dates)
        t = np.arange(len(dates))
        for i in range(1, n_terms + 1):
            df[f"fourier_sin_{period}_{i}"] = np.sin(2 * np.pi * i * t / period)
            df[f"fourier_cos_{period}_{i}"] = np.cos(2 * np.pi * i * t / period)
        return df

    def create_future_features(
        self,
        index,
        future_dates: pd.DatetimeIndex,
        series: pd.Series = None,
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """Create features for future dates (for forecasting)"""
        if feature_names is None:
            feature_names = self.feature_names

        # Create a dummy series for feature generation
        size = len(future_dates)
        dummy_series = pd.Series(
            [np.nan] * (size),
            index=pd.Index(future_dates, name="Month"),
            name=series.name,
        )
        features = self.create_features(
            pd.concat([series, dummy_series]), future_dates
        )[-size:]

        # Select only the features we have names for
        if feature_names:
            self.feature_names = feature_names
            available_features = [f for f in feature_names if f in features.columns]
            return features[available_features]

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names"""
        return self.feature_names.copy()
