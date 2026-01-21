import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class UserBehaviorFeatureEngineer:
    """
    Modular feature engineering pipeline for user behavior time series analysis.
    """

    def __init__(self, window_size: int = 60, default_alpha: float = 0.3):
        """
        Initialize the feature engineer.

        Parameters:
        -----------
        window_size : int
            Rolling window size in minutes for statistical estimation
        default_alpha : float
            Smoothing parameter for exponential decay in probability estimation
        """
        self.window_size = window_size
        self.default_alpha = default_alpha
        self.user_history = {}

    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess the CSV data.

        Parameters:
        -----------
        filepath : str
            Path to the CSV file

        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame with datetime indices
        """
        df = pd.read_csv(filepath)

        # Convert timestamps
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
        df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])

        # Calculate session duration
        df["session_duration"] = (
            df["end_timestamp"] - df["start_timestamp"]
        ).dt.total_seconds() / 60

        # Sort by start time
        df = df.sort_values("start_timestamp").reset_index(drop=True)

        return df

    def create_minute_series(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create aggregated time series with 1-minute intervals.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed DataFrame

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (aggregated_series, user_activity_matrix)
        """
        # Initialize empty DataFrames
        aggregated_series = pd.DataFrame()
        aggregated_series["volume"] = df["usage_volume"]
        aggregated_series["end_timestamp"] = df["end_timestamp"]
        aggregated_series = aggregated_series.sort_values("end_timestamp")
        aggregated_series = (
            aggregated_series.set_index("end_timestamp")
            .resample("1min")["volume"]
            .sum()
            .reset_index()
        )
        user_activity = pd.DataFrame(
            0, index=aggregated_series.index, columns=sorted(df["user_id"].unique())
        )

        # Aggregate volume and track user activity per minute
        active_counts = []
        for minute in aggregated_series["end_timestamp"]:
            next_minute = minute + pd.Timedelta(minutes=1)

            # Find sessions active during this minute
            mask = (df["start_timestamp"] < next_minute) & (
                df["end_timestamp"] > minute
            )
            active_sessions = df[mask]

            active_counts.append(len(active_sessions["user_id"].unique()))

            # Update user activity matrix
            active_users = active_sessions["user_id"].unique()
            user_activity.loc[minute, active_users] = 1

        aggregated_series["active_users"] = active_counts

        aggregated_series = aggregated_series.set_index("end_timestamp")

        return aggregated_series, user_activity


    def create_minute_series(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Vectorized version with uniform consumption assumption.
        """
        # Create time range covering all data
        start_time = df["start_timestamp"].min().floor("min")
        end_time = df["end_timestamp"].max().ceil("min")

        # Create complete minute index
        full_minute_range = pd.date_range(
            start=start_time,
            end=end_time,
            freq="1min",
            inclusive="left"
        )

        # Initialize result structures
        volume_series = pd.Series(0, index=pd.DatetimeIndex(full_minute_range))
        user_activity_dict = {user_id: pd.Series(0, index=full_minute_range)
                             for user_id in sorted(df["user_id"].unique())}

        # Process each session
        for _, row in df.iterrows():
            session_start = row["start_timestamp"]
            session_end = row["end_timestamp"]
            session_volume = row["usage_volume"]
            user_id = row["user_id"]

            # Skip invalid sessions
            if session_end <= session_start:
                continue

            # Get overlapping minutes
            first_minute = session_start.floor("min")
            last_minute = session_end.ceil("min") - pd.Timedelta(minutes=1)

            # Create minute intervals for this session
            session_minutes = pd.date_range(
                start=first_minute,
                end=last_minute + pd.Timedelta(minutes=1),  # Add 1 to include last minute
                freq="1min",
                inclusive="left"
            )

            # Calculate volume for each minute
            for minute in session_minutes:
                minute_start = minute
                minute_end = minute + pd.Timedelta(minutes=1)

                # Calculate overlap
                overlap_start = max(session_start, minute_start)
                overlap_end = min(session_end, minute_end)
                overlap_duration = max(0, (overlap_end - overlap_start).total_seconds())
                total_duration = (session_end - session_start).total_seconds()

                if overlap_duration > 0:
                    minute_volume = session_volume * (overlap_duration / total_duration)
                    volume_series.loc[minute] += minute_volume
                    user_activity_dict[user_id].loc[minute] = 1

        # Create final DataFrames
        aggregated_series = pd.DataFrame({
            "end_timestamp": volume_series.index,
            "volume": volume_series.values,
            "active_users": pd.DataFrame(user_activity_dict).sum(axis=1).values
        }, index=volume_series.index)

        user_activity = pd.DataFrame(user_activity_dict, index=volume_series.index)
        return aggregated_series, user_activity

    def estimate_survival_probability(
        self, user_activity: pd.DataFrame, user_id: str, current_time: pd.Timestamp
    ) -> float:
        """
        Estimate survival probability using Exponential or Weibull distribution.

        Parameters:
        -----------
        user_activity : pd.DataFrame
            User activity matrix
        user_id : str
            User identifier
        current_time : pd.Timestamp
            Current time for probability estimation

        Returns:
        --------
        float
            Survival probability for next minute
        """
        # Get user's activity history
        user_history = user_activity[user_id]

        # Get last window_size minutes of history
        window_start = current_time - pd.Timedelta(minutes=self.window_size)
        if window_start not in user_history:
            return 0.5

        history_window = user_history.loc[window_start:current_time]

        # Calculate session gaps (times between activities)
        activity_times = history_window[history_window == 1].index
        if len(activity_times) < 2:
            return 0.5  # Default probability if insufficient data

        gaps = []
        for i in range(1, len(activity_times)):
            gap = (activity_times[i] - activity_times[i - 1]).total_seconds() / 60
            gaps.append(gap)

        if not gaps:
            return 0.5

        # Fit exponential distribution to gaps (simplest survival model)
        try:
            # Estimate lambda parameter for exponential distribution
            lambda_param = 1 / np.mean(gaps)

            # Survival probability for next minute
            survival_prob = np.exp(-lambda_param * 1)  # P(survive > 1 minute)
            return min(max(survival_prob, 0.01), 0.99)  # Bound probabilities

        except:
            return 0.5

    def estimate_default_probability(
        self, user_activity: pd.DataFrame, current_time: pd.Timestamp
    ) -> Dict:
        """
        Estimate probability that any user will default (stop consuming) in next minute.

        Parameters:
        -----------
        user_activity : pd.DataFrame
            User activity matrix
        current_time : pd.Timestamp
            Current time for estimation

        Returns:
        --------
        Dict
            Dictionary with default probabilities and statistics
        """
        # Get currently active users
        current_active = user_activity.columns[
            user_activity.loc[current_time] == 1
        ].tolist()

        if not current_active:
            return {
                "expected_defaults": 0,
                "default_probabilities": {},
                "active_users_count": 0,
            }

        # Calculate survival probabilities for each active user
        survival_probs = {}
        for user in current_active:
            survival_probs[user] = self.estimate_survival_probability(
                user_activity, user, current_time
            )

        # Convert to default probabilities
        default_probs = {user: 1 - prob for user, prob in survival_probs.items()}

        # Expected number of defaults (sum of probabilities)
        expected_defaults = sum(default_probs.values())

        return {
            "expected_defaults": expected_defaults,
            "default_probabilities": default_probs,
            "active_users_count": len(current_active),
            "avg_default_prob": np.mean(list(default_probs.values())),
        }

    def calculate_session_features(
        self, df: pd.DataFrame, time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Calculate session-based features.

        Parameters:
        -----------
        df : pd.DataFrame
            Original session data
        time_index : pd.DatetimeIndex
            Time index for the aggregated series

        Returns:
        --------
        pd.DataFrame
            DataFrame with session features
        """
        features = pd.DataFrame(index=time_index)

        for minute in time_index:
            next_minute = minute + pd.Timedelta(minutes=1)

            # Sessions starting in this minute
            new_sessions = df[
                (df["start_timestamp"] >= minute)
                & (df["start_timestamp"] < next_minute)
            ]

            # Sessions ending in this minute
            ending_sessions = df[
                (df["end_timestamp"] >= minute) & (df["end_timestamp"] < next_minute)
            ]

            # Calculate features
            features.loc[minute, "new_sessions"] = len(new_sessions)
            features.loc[minute, "ending_sessions"] = len(ending_sessions)

            if len(new_sessions) > 0:
                features.loc[minute, "avg_new_session_volume"] = new_sessions[
                    "usage_volume"
                ].mean()
                features.loc[minute, "avg_new_session_duration"] = new_sessions[
                    "session_duration"
                ].mean()
            else:
                features.loc[minute, "avg_new_session_volume"] = 0
                features.loc[minute, "avg_new_session_duration"] = 0

        return features

    def calculate_rolling_features(
        self, aggregated_series: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling statistical features.

        Parameters:
        -----------
        aggregated_series : pd.DataFrame
            Aggregated time series

        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=aggregated_series.index)

        # Rolling statistics for volume
        features["volume_rolling_mean"] = (
            aggregated_series["volume"]
            .rolling(window=self.window_size, min_periods=1)
            .mean()
        )
        features["volume_rolling_std"] = (
            aggregated_series["volume"]
            .rolling(window=self.window_size, min_periods=1)
            .std()
        )
        features["volume_z_score"] = (
            aggregated_series["volume"] - features["volume_rolling_mean"]
        ) / features["volume_rolling_std"].replace(0, 1)

        # Exponential moving averages
        features["volume_ema"] = (
            aggregated_series["volume"]
            .ewm(alpha=self.default_alpha, adjust=False)
            .mean()
        )

        # Rolling statistics for active users
        features["users_rolling_mean"] = (
            aggregated_series["active_users"]
            .rolling(window=self.window_size, min_periods=1)
            .mean()
        )
        features["users_rolling_std"] = (
            aggregated_series["active_users"]
            .rolling(window=self.window_size, min_periods=1)
            .std()
        )

        # Rate of change features
        features["volume_change"] = aggregated_series["volume"].pct_change()
        features["users_change"] = aggregated_series["active_users"].pct_change()

        # Volatility features
        features["volume_volatility"] = (
            aggregated_series["volume"].rolling(window=30, min_periods=1).std()
            / aggregated_series["volume"].rolling(window=30, min_periods=1).mean()
        )

        return features

    def calculate_time_features(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate time-based features.

        Parameters:
        -----------
        time_index : pd.DatetimeIndex
            Time index

        Returns:
        --------
        pd.DataFrame
            DataFrame with time features
        """
        features = pd.DataFrame(index=time_index)

        features["hour"] = time_index.hour
        features["minute_of_day"] = time_index.hour * 60 + time_index.minute
        features["day_of_week"] = time_index.dayofweek
        features["is_weekend"] = (time_index.dayofweek >= 5).astype(int)
        features["day_of_month"] = time_index.day
        features["month"] = time_index.month

        # Cyclical encoding for hour
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

        return features

    def engineer_features(
        self, filepath: str, include_default_prediction: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        include_default_prediction : bool
            Whether to include default prediction features

        Returns:
        --------
        pd.DataFrame
            DataFrame with all engineered features
        """
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess(filepath)

        print("Creating aggregated time series...")
        aggregated_series, user_activity = self.create_minute_series(df)

        # Initialize result DataFrame with basic aggregates
        result = aggregated_series.copy()

        # print("Calculating time-based features...")
        time_features = self.calculate_time_features(result.index)
        result = pd.concat([result, time_features], axis=1)

        print("Calculating rolling statistical features...")
        rolling_features = self.calculate_rolling_features(aggregated_series)
        result = pd.concat([result, rolling_features], axis=1)

        print("Calculating session-based features...")
        session_features = self.calculate_session_features(df, result.index)
        result = pd.concat([result, session_features], axis=1)

        if include_default_prediction:
            print("Estimating default probabilities...")
            default_estimates = []

            for idx, minute in enumerate(result.index):
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(result)} minutes...")

                estimate = self.estimate_default_probability(user_activity, minute)
                default_estimates.append(estimate["expected_defaults"])

            result["expected_defaults_next_minute"] = default_estimates

            # Calculate default rate features
            result["default_rate"] = result["expected_defaults_next_minute"] / result[
                "active_users"
            ].replace(0, 1)
            result["default_rate_ema"] = (
                result["default_rate"]
                .ewm(alpha=self.default_alpha, adjust=False)
                .mean()
            )

        print("Feature engineering complete!")
        print(f"Total features: {len(result.columns)}")
        print(f"Time range: {result.index[0]} to {result.index[-1]}")

        return result

    def get_feature_descriptions(self) -> Dict:
        """
        Get descriptions of all engineered features.

        Returns:
        --------
        Dict
            Dictionary with feature descriptions
        """
        descriptions = {
            "volume": "Aggregated usage volume per minute",
            "active_users": "Number of active users in the minute",
            "hour": "Hour of day",
            "minute_of_day": "Minute of day (0-1439)",
            "day_of_week": "Day of week (0=Monday)",
            "is_weekend": "Whether it's weekend (1) or not (0)",
            "day_of_month": "Day of month",
            "month": "Month",
            "hour_sin": "Cyclical encoding of hour (sine)",
            "hour_cos": "Cyclical encoding of hour (cosine)",
            "volume_rolling_mean": f"Rolling mean of volume ({self.window_size}min window)",
            "volume_rolling_std": f"Rolling std of volume ({self.window_size}min window)",
            "volume_z_score": "Z-score of volume relative to rolling window",
            "volume_ema": f"Exponential moving average of volume (alpha={self.default_alpha})",
            "users_rolling_mean": f"Rolling mean of active users ({self.window_size}min window)",
            "users_rolling_std": f"Rolling std of active users ({self.window_size}min window)",
            "volume_change": "Percentage change in volume from previous minute",
            "users_change": "Percentage change in active users from previous minute",
            "volume_volatility": "Rolling coefficient of variation for volume (30min window)",
            "new_sessions": "Number of new sessions starting in the minute",
            "ending_sessions": "Number of sessions ending in the minute",
            "avg_new_session_volume": "Average volume of new sessions starting",
            "avg_new_session_duration": "Average duration of new sessions starting",
            "expected_defaults_next_minute": "Expected number of users who will stop consuming in next minute",
            "default_rate": "Expected default rate (expected defaults / active users)",
            "default_rate_ema": "Exponential moving average of default rate",
        }

        return descriptions


# Example usage and statistical validation
def validate_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate engineered features through statistical analysis.

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with engineered features

    Returns:
    --------
    pd.DataFrame
        Statistical validation results
    """
    validation_results = []

    # Target variable for correlation analysis (future volume)
    features_df["volume_next_minute"] = features_df["volume"].shift(-1)

    # Remove NaN values for correlation
    valid_features = features_df.dropna()

    for col in valid_features.columns:
        if col not in ["volume_next_minute", "volume", "active_users"]:
            try:
                # Calculate correlation with future volume
                corr = valid_features[col].corr(valid_features["volume_next_minute"])

                # Calculate information value (simplified)
                n_bins = min(20, len(valid_features[col].unique()))
                if n_bins > 1:
                    # Discretize for mutual information estimation
                    x_binned = pd.qcut(valid_features[col], n_bins, duplicates="drop")
                    y_binned = pd.qcut(
                        valid_features["volume_next_minute"], n_bins, duplicates="drop"
                    )

                    # Calculate normalized mutual information
                    contingency = pd.crosstab(x_binned, y_binned, normalize=True)
                    mi = 0
                    for i in range(len(contingency.index)):
                        for j in range(len(contingency.columns)):
                            p_xy = contingency.iloc[i, j]
                            p_x = contingency.iloc[:, j].sum()
                            p_y = contingency.iloc[i, :].sum()
                            if p_xy > 0 and p_x > 0 and p_y > 0:
                                mi += p_xy * np.log(p_xy / (p_x * p_y))

                    mi_normalized = mi / np.log(n_bins)
                else:
                    mi_normalized = 0

                # Calculate feature importance via variance
                variance_ratio = (
                    valid_features[col].var() / valid_features["volume"].var()
                )

                validation_results.append(
                    {
                        "feature": col,
                        "correlation_with_next_volume": corr,
                        "normalized_mutual_info": mi_normalized,
                        "variance_ratio": variance_ratio,
                        "mean": valid_features[col].mean(),
                        "std": valid_features[col].std(),
                        "has_nan": features_df[col].isna().sum(),
                    }
                )

            except:
                continue

    return pd.DataFrame(validation_results).sort_values(
        "correlation_with_next_volume", key=abs, ascending=False
    )


# Main execution example
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = UserBehaviorFeatureEngineer(window_size=60, default_alpha=0.3)

    # Run feature engineering pipeline
    features = engineer.engineer_features("data.csv", include_default_prediction=True)

    # Save features
    features.to_csv("engineered_features.csv")

    # Validate features
    validation_results = validate_features(features)
    print("\nTop 10 most predictive features:")
    print(validation_results.head(10))

    # Save validation results
    validation_results.to_csv("feature_validation.csv", index=False)

    # Get feature descriptions
    descriptions = engineer.get_feature_descriptions()
    pd.Series(descriptions).to_csv("feature_descriptions.csv")

    print("\nFeature engineering complete!")
    print(f"Generated {len(features.columns)} features")
    print(f"Time series length: {len(features)} minutes")
