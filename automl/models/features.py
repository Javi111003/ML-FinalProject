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
        series: pd.DataFrame,
        dates: pd.DatetimeIndex,
        feature_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create all configured features"""
        if feature_types is None:
            feature_types = ["lag", "rolling", "seasonal", "fourier"]

        feature_df = series.copy()

        features_list = [self.create_timestamps(feature_df, dates)]

        # Lag features
        if "lag" in feature_types:
            lag_config = self.config.get("lag_features", {})
            max_lags = lag_config.get("max_lags", 3)
            lag_features = self.create_lag_features(feature_df, series, max_lags)
            features_list.append(lag_features)

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
        return feature_df

    @staticmethod
    def create_lag_features(df, series: pd.Series, max_lags: int = 3) -> pd.DataFrame:
        """Create lag features for time series"""
        for lag in range(30, max_lags + 1, 30):

            df[f"lag_{lag}"] = series.shift(lag)
        return df

    @staticmethod
    def create_rolling_features(
        df, series: pd.Series, windows: List[int] = None, stats: List[str] = None
    ) -> pd.DataFrame:
        """Create rolling statistics features"""
        if windows is None:
            windows = [30, 60, 90]
        if stats is None:
            stats = ["mean", "std", "min", "max"]
        shifted_series = series.shift(1)

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
        data = {col: [0]*size for col in series}
        dummy_series = pd.DataFrame(data, index=future_dates)

        combined = pd.concat([series, dummy_series])
        combined.index.freq = future_dates.freq
        df = self.create_features(
            combined, combined.index
        )
        del df[df.columns[0]]

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
