import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

from users import UserBehaviorFeatureEngineer

warnings.filterwarnings("ignore")


class UserClusteringEngine:
    """
    Modular user clustering system based on consumption patterns.
    Creates 9 clusters (3x3 matrix) based on:
    - Session duration: low, medium, high
    - Consumption volume: low, medium, high
    """

    def __init__(
        self,
        duration_percentiles: Tuple[float, float] = (33, 66),
        volume_percentiles: Tuple[float, float] = (33, 66),
    ):
        """
        Initialize the clustering engine.

        Parameters:
        -----------
        duration_percentiles : Tuple[float, float]
            Percentiles for low/medium/high duration categorization
        volume_percentiles : Tuple[float, float]
            Percentiles for low/medium/high volume categorization
        """
        self.duration_percentiles = duration_percentiles
        self.volume_percentiles = volume_percentiles
        self.user_clusters = None
        self.cluster_stats = None
        self.cluster_descriptions = None

    def extract_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user-level features for clustering.

        Parameters:
        -----------
        df : pd.DataFrame
            Original session data with columns:
            start_timestamp, end_timestamp, user_id, usage_volume

        Returns:
        --------
        pd.DataFrame
            User-level features DataFrame
        """
        # Calculate session-level features
        df = df.copy()
        df["session_duration"] = (
            df["end_timestamp"] - df["start_timestamp"]
        ).dt.total_seconds() / 60

        # Group by user
        user_features = (
            df.groupby("user_id")
            .agg(
                {
                    "usage_volume": ["sum", "mean", "std", "count"],
                    "session_duration": ["mean", "sum", "std", "max"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        user_features.columns = [
            "user_id",
            "total_volume",
            "avg_volume_per_session",
            "std_volume",
            "session_count",
            "avg_duration",
            "total_duration",
            "std_duration",
            "max_duration",
        ]

        # Additional derived features
        user_features["volume_per_minute"] = user_features[
            "total_volume"
        ] / user_features["total_duration"].replace(0, np.nan)
        user_features["sessions_per_day"] = user_features["session_count"] / (
            (df["end_timestamp"].max() - df["start_timestamp"].min()).days + 1
        )
        user_features["consistency_score"] = 1 / (
            1 + user_features["std_volume"].fillna(0)
        )

        # Handle infinite values
        user_features = user_features.replace([np.inf, -np.inf], np.nan)

        return user_features

    def categorize_users_percentile(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize users into 9 clusters using percentile-based thresholds.

        Parameters:
        -----------
        user_features : pd.DataFrame
            User-level features DataFrame

        Returns:
        --------
        pd.DataFrame
            DataFrame with cluster assignments
        """
        # Calculate percentile thresholds
        duration_low_thresh = np.percentile(
            user_features["avg_duration"].dropna(), self.duration_percentiles[0]
        )
        duration_high_thresh = np.percentile(
            user_features["avg_duration"].dropna(), self.duration_percentiles[1]
        )

        volume_low_thresh = np.percentile(
            user_features["avg_volume_per_session"].dropna(), self.volume_percentiles[0]
        )
        volume_high_thresh = np.percentile(
            user_features["avg_volume_per_session"].dropna(), self.volume_percentiles[1]
        )

        # Categorize duration
        def categorize_duration(duration):
            if pd.isna(duration):
                return "unknown"
            elif duration < duration_low_thresh:
                return "low"
            elif duration < duration_high_thresh:
                return "medium"
            else:
                return "high"

        # Categorize volume
        def categorize_volume(volume):
            if pd.isna(volume):
                return "unknown"
            elif volume < volume_low_thresh:
                return "low"
            elif volume < volume_high_thresh:
                return "medium"
            else:
                return "high"

        # Apply categorization
        user_features["duration_category"] = user_features["avg_duration"].apply(
            categorize_duration
        )
        user_features["volume_category"] = user_features[
            "avg_volume_per_session"
        ].apply(categorize_volume)

        # Create 9 clusters (3x3 matrix)
        category_to_id = {
            ("low", "low"): 0,
            ("low", "medium"): 1,
            ("low", "high"): 2,
            ("medium", "low"): 3,
            ("medium", "medium"): 4,
            ("medium", "high"): 5,
            ("high", "low"): 6,
            ("high", "medium"): 7,
            ("high", "high"): 8,
        }

        user_features["cluster_id"] = user_features.apply(
            lambda row: category_to_id.get(
                (row["duration_category"], row["volume_category"]), -1
            ),
            axis=1,
        )

        # Store thresholds
        self.thresholds = {
            "duration": {"low": duration_low_thresh, "high": duration_high_thresh},
            "volume": {"low": volume_low_thresh, "high": volume_high_thresh},
        }

        return user_features

    def categorize_users_kmeans(
        self, user_features: pd.DataFrame, n_clusters: int = 9
    ) -> pd.DataFrame:
        """
        Categorize users using KMeans clustering and then map to 3x3 grid.

        Parameters:
        -----------
        user_features : pd.DataFrame
            User-level features DataFrame
        n_clusters : int
            Number of clusters to create (should be 9 for 3x3 grid)

        Returns:
        --------
        pd.DataFrame
            DataFrame with cluster assignments
        """
        # Select features for clustering
        clustering_features = [
            "avg_duration",
            "avg_volume_per_session",
            "total_volume",
            "session_count",
        ]

        # Prepare data
        X = user_features[clustering_features].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_features["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

        # Map KMeans clusters to 3x3 grid based on centroids
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=clustering_features)

        # Calculate percentiles for centroids
        centroids_df["duration_percentile"] = centroids_df["avg_duration"].rank(
            pct=True
        )
        centroids_df["volume_percentile"] = centroids_df["avg_volume_per_session"].rank(
            pct=True
        )

        # Map to 3x3 grid
        def map_to_grid(duration_pct, volume_pct):
            if duration_pct < 0.33:
                duration_cat = "low"
            elif duration_pct < 0.66:
                duration_cat = "medium"
            else:
                duration_cat = "high"

            if volume_pct < 0.33:
                volume_cat = "low"
            elif volume_pct < 0.66:
                volume_cat = "medium"
            else:
                volume_cat = "high"

            category_to_id = {
                ("low", "low"): 0,
                ("low", "medium"): 1,
                ("low", "high"): 2,
                ("medium", "low"): 3,
                ("medium", "medium"): 4,
                ("medium", "high"): 5,
                ("high", "low"): 6,
                ("high", "medium"): 7,
                ("high", "high"): 8,
            }

            return category_to_id.get((duration_cat, volume_cat), -1)

        cluster_mapping = {}
        for cluster_idx in range(n_clusters):
            duration_pct = centroids_df.loc[cluster_idx, "duration_percentile"]
            volume_pct = centroids_df.loc[cluster_idx, "volume_percentile"]
            grid_id = map_to_grid(duration_pct, volume_pct)
            cluster_mapping[cluster_idx] = grid_id

        # Apply mapping
        user_features["cluster_id"] = user_features["kmeans_cluster"].map(
            cluster_mapping
        )

        # For users that didn't get mapped properly, use percentile-based categorization
        mask = user_features["cluster_id"].isna() | (user_features["cluster_id"] == -1)
        if mask.any():
            backup_categorized = self.categorize_users_percentile(user_features[mask])
            user_features.loc[mask, "cluster_id"] = backup_categorized["cluster_id"]

        return user_features

    def analyze_clusters(self, user_features: pd.DataFrame) -> Dict:
        """
        Analyze and describe each cluster.

        Parameters:
        -----------
        user_features : pd.DataFrame
            DataFrame with cluster assignments

        Returns:
        --------
        Dict
            Dictionary with cluster statistics and descriptions
        """
        cluster_stats = (
            user_features.groupby("cluster_id")
            .agg(
                {
                    "user_id": "count",
                    "avg_duration": ["mean", "std", "min", "max"],
                    "avg_volume_per_session": ["mean", "std", "min", "max"],
                    "total_volume": "mean",
                    "session_count": "mean",
                    "volume_per_minute": "mean",
                }
            )
            .round(2)
        )

        # Flatten column names
        cluster_stats.columns = [
            "_".join(col).strip() for col in cluster_stats.columns.values
        ]
        cluster_stats = cluster_stats.rename(columns={"user_id_count": "user_count"})

        # Create human-readable descriptions
        cluster_descriptions = {}
        cluster_grid = {
            0: "Low Duration, Low Volume - Casual users",
            1: "Low Duration, Medium Volume - Efficient users",
            2: "Low Duration, High Volume - Power users (brief but intense)",
            3: "Medium Duration, Low Volume - Extended casual users",
            4: "Medium Duration, Medium Volume - Regular users",
            5: "Medium Duration, High Volume - High-value regular users",
            6: "High Duration, Low Volume - Light but persistent users",
            7: "High Duration, Medium Volume - Engaged users",
            8: "High Duration, High Volume - Heavy users",
        }

        for cluster_id, description in cluster_grid.items():
            if cluster_id in cluster_stats.index:
                stats = cluster_stats.loc[cluster_id]
                cluster_descriptions[cluster_id] = {
                    "name": description,
                    "user_count": int(stats["user_count"]),
                    "avg_duration": stats["avg_duration_mean"],
                    "avg_volume": stats["avg_volume_per_session_mean"],
                    "avg_total_volume": stats["total_volume_mean"],
                    "avg_sessions": stats["session_count_mean"],
                    "volume_per_minute": stats["volume_per_minute_mean"],
                }

        self.cluster_stats = cluster_stats
        self.cluster_descriptions = cluster_descriptions

        return {"stats": cluster_stats, "descriptions": cluster_descriptions}

    def fit(
        self, df: pd.DataFrame, method: str = "percentile"
    ) -> "UserClusteringEngine":
        """
        Fit clustering model to data.

        Parameters:
        -----------
        df : pd.DataFrame
            Original session data
        method : str
            Clustering method: 'percentile' or 'kmeans'

        Returns:
        --------
        UserClusteringEngine
            Fitted clustering engine
        """
        # Extract user features
        user_features = self.extract_user_features(df)

        # Apply clustering method
        if method == "kmeans":
            user_features = self.categorize_users_kmeans(user_features)
        else:  # percentile method
            user_features = self.categorize_users_percentile(user_features)

        # Store user clusters
        self.user_clusters = user_features[
            ["user_id", "cluster_id", "duration_category", "volume_category"]
        ]

        # Analyze clusters
        self.analyze_clusters(user_features)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster IDs to session data.

        Parameters:
        -----------
        df : pd.DataFrame
            Original session data

        Returns:
        --------
        pd.DataFrame
            DataFrame with cluster_id column
        """
        if self.user_clusters is None:
            raise ValueError("Clustering engine not fitted. Call fit() first.")

        df = df.copy()

        # Map user IDs to cluster IDs
        cluster_map = self.user_clusters.set_index("user_id")["cluster_id"].to_dict()
        df["cluster_id"] = df["user_id"].map(cluster_map)

        # For users not in clustering (new users), assign to most similar cluster
        # based on average values
        missing_mask = df["cluster_id"].isna()
        if missing_mask.any():
            # Calculate average for new users and assign to nearest cluster centroid
            new_users = df[missing_mask].copy()
            new_users["session_duration"] = (
                new_users["end_timestamp"] - new_users["start_timestamp"]
            ).dt.total_seconds() / 60

            # Group new users and calculate features
            new_user_stats = (
                new_users.groupby("user_id")
                .agg({"usage_volume": "mean", "session_duration": "mean"})
                .reset_index()
            )

            # Simple assignment based on thresholds
            for _, row in new_user_stats.iterrows():
                duration_cat = "medium"
                volume_cat = "medium"

                # Duration categorization
                if row["session_duration"] < self.thresholds["duration"]["low"]:
                    duration_cat = "low"
                elif row["session_duration"] > self.thresholds["duration"]["high"]:
                    duration_cat = "high"

                # Volume categorization
                if row["usage_volume"] < self.thresholds["volume"]["low"]:
                    volume_cat = "low"
                elif row["usage_volume"] > self.thresholds["volume"]["high"]:
                    volume_cat = "high"

                # Map to cluster ID
                category_to_id = {
                    ("low", "low"): 0,
                    ("low", "medium"): 1,
                    ("low", "high"): 2,
                    ("medium", "low"): 3,
                    ("medium", "medium"): 4,
                    ("medium", "high"): 5,
                    ("high", "low"): 6,
                    ("high", "medium"): 7,
                    ("high", "high"): 8,
                }

                cluster_id = category_to_id.get(
                    (duration_cat, volume_cat), 4
                )  # Default to medium-medium

                # Update DataFrame
                df.loc[df["user_id"] == row["user_id"], "cluster_id"] = cluster_id

        return df

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary of all clusters.

        Returns:
        --------
        pd.DataFrame
            DataFrame with cluster summaries
        """
        if self.cluster_descriptions is None:
            raise ValueError("No cluster analysis available. Run fit() first.")

        summary_data = []
        for cluster_id, desc in self.cluster_descriptions.items():
            summary_data.append(
                {
                    "cluster_id": cluster_id,
                    "name": desc["name"],
                    "user_count": desc["user_count"],
                    "avg_duration": desc["avg_duration"],
                    "avg_volume": desc["avg_volume"],
                    "avg_total_volume": desc["avg_total_volume"],
                    "avg_sessions": desc["avg_sessions"],
                    "volume_per_minute": desc["volume_per_minute"],
                }
            )

        return pd.DataFrame(summary_data).sort_values("user_count", ascending=False)


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer that supports both user-level and cluster-level analysis.
    """

    def __init__(self, base_engineer, clustering_engine):
        """
        Initialize enhanced feature engineer.

        Parameters:
        -----------
        base_engineer : UserBehaviorFeatureEngineer
            Base feature engineering engine
        clustering_engine : UserClusteringEngine
            Clustering engine
        """
        self.base_engineer = base_engineer
        self.clustering_engine = clustering_engine
        self.cluster_analyses = {}

    def analyze_all_users(self, filepath: str) -> pd.DataFrame:
        """
        Analyze all users together (original functionality).

        Parameters:
        -----------
        filepath : str
            Path to CSV file

        Returns:
        --------
        pd.DataFrame
            Feature-engineered DataFrame for all users
        """
        return self.base_engineer.engineer_features(
            filepath, include_default_prediction=True
        )

    def analyze_by_cluster(
        self, filepath: str, cluster_id: Optional[int] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Analyze users by cluster, either specific cluster or all clusters.

        Parameters:
        -----------
        filepath : str
            Path to CSV file
        cluster_id : Optional[int]
            Specific cluster ID to analyze, or None for all clusters

        Returns:
        --------
        Dict[int, pd.DataFrame]
            Dictionary mapping cluster IDs to feature-engineered DataFrames
        """
        # Load and preprocess data
        df = self.base_engineer.load_and_preprocess(filepath)

        # Add cluster IDs
        df_with_clusters = self.clustering_engine.transform(df)

        # Determine which clusters to analyze
        if cluster_id is not None:
            cluster_ids = [cluster_id]
        else:
            cluster_ids = sorted(df_with_clusters["cluster_id"].unique())

        results = {}

        for cid in cluster_ids:
            print(
                f"\nAnalyzing Cluster {cid}: {self.clustering_engine.cluster_descriptions[cid]['name']}"
            )

            # Filter data for this cluster
            cluster_data = df_with_clusters[
                df_with_clusters["cluster_id"] == cid
            ].copy()

            # Replace user_id with cluster_id for aggregation
            # This treats the entire cluster as a single "user" for time series analysis
            cluster_data["original_user_id"] = cluster_data["user_id"]
            cluster_data["user_id"] = cid  # Replace with cluster ID

            # Create time series for this cluster
            aggregated_series, user_activity = self.base_engineer.create_minute_series(
                cluster_data
            )

            # Initialize result DataFrame
            result = aggregated_series.copy()

            # Add time features
            time_features = self.base_engineer.calculate_time_features(result.index)
            result = pd.concat([result, time_features], axis=1)

            # Add rolling features
            rolling_features = self.base_engineer.calculate_rolling_features(
                aggregated_series
            )
            result = pd.concat([result, rolling_features], axis=1)

            # Add session features
            session_features = self.base_engineer.calculate_session_features(
                cluster_data, result.index
            )
            result = pd.concat([result, session_features], axis=1)

            # Add default prediction (for the cluster as a whole)
            default_estimates = []
            for idx, minute in enumerate(result.index):
                estimate = self.base_engineer.estimate_default_probability(
                    user_activity, minute
                )
                default_estimates.append(estimate["expected_defaults"])

            result["expected_defaults_next_minute"] = default_estimates
            result["default_rate"] = result["expected_defaults_next_minute"] / result[
                "active_users"
            ].replace(0, 1)
            result["default_rate_ema"] = (
                result["default_rate"]
                .ewm(alpha=self.base_engineer.default_alpha, adjust=False)
                .mean()
            )

            # Add cluster metadata
            result["cluster_id"] = cid
            result["cluster_name"] = self.clustering_engine.cluster_descriptions[cid][
                "name"
            ]

            results[cid] = result

            # Store in cache
            self.cluster_analyses[cid] = result

            print(
                f"  - Cluster size: {self.clustering_engine.cluster_descriptions[cid]['user_count']} users"
            )
            print(f"  - Time series length: {len(result)} minutes")
            print(f"  - Average volume per minute: {result['volume'].mean():.2f}")
            print(
                f"  - Average active users per minute: {result['active_users'].mean():.2f}"
            )

        return results

    def compare_clusters(
        self, cluster_results: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare metrics across different clusters.

        Parameters:
        -----------
        cluster_results : Dict[int, pd.DataFrame]
            Dictionary of cluster analysis results

        Returns:
        --------
        pd.DataFrame
            Comparison metrics across clusters
        """
        comparison_data = []

        for cluster_id, df in cluster_results.items():
            # Calculate key metrics
            metrics = {
                "cluster_id": cluster_id,
                "cluster_name": self.clustering_engine.cluster_descriptions[cluster_id][
                    "name"
                ],
                "total_volume": df["volume"].sum(),
                "avg_volume_per_minute": df["volume"].mean(),
                "avg_active_users": df["active_users"].mean(),
                "avg_expected_defaults": df["expected_defaults_next_minute"].mean(),
                "avg_default_rate": df["default_rate"].mean(),
                "volume_volatility": df["volume_volatility"].mean(),
                "peak_hour": df.groupby("hour")["volume"].mean().idxmax(),
                "peak_volume": df.groupby("hour")["volume"].mean().max(),
                "new_sessions_per_minute": df["new_sessions"].mean(),
                "session_duration_avg": df["avg_new_session_duration"].mean(),
            }

            # Calculate time-based patterns
            weekday_volume = df[df["is_weekend"] == 0]["volume"].mean()
            weekend_volume = df[df["is_weekend"] == 1]["volume"].mean()
            metrics["weekday_weekend_ratio"] = (
                weekday_volume / weekend_volume if weekend_volume > 0 else np.nan
            )

            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)

        # Add normalized metrics
        for col in ["total_volume", "avg_volume_per_minute", "avg_active_users"]:
            comparison_df[f"{col}_normalized"] = (
                comparison_df[col] / comparison_df[col].sum()
            )

        return comparison_df.sort_values("total_volume", ascending=False)

    def create_cluster_timeseries(self, filepath: str) -> pd.DataFrame:
        """
        Create a unified time series with cluster-level aggregation.

        Parameters:
        -----------
        filepath : str
            Path to CSV file

        Returns:
        --------
        pd.DataFrame
            Unified time series with cluster-level columns
        """
        # Load and preprocess data
        df = self.base_engineer.load_and_preprocess(filepath)

        # Add cluster IDs
        df_with_clusters = self.clustering_engine.transform(df)

        # Create time index
        start_time = df_with_clusters["start_timestamp"].min().floor("min")
        end_time = df_with_clusters["end_timestamp"].max().ceil("min")
        time_index = pd.date_range(start=start_time, end=end_time, freq="1min")

        # Add total metrics (all users)
        unified_series = self.base_engineer.engineer_features(filepath)

        # Add cluster-specific metrics
        for cluster_id in range(9):  # All 9 clusters
            name = f"{cluster_id}_{filepath}"
            if not Path(name).exists():
                cluster_data = df_with_clusters[
                    df_with_clusters["cluster_id"] == cluster_id
                ].copy()
                cluster_data.to_csv(name)
            cluster_df = self.base_engineer.engineer_features(name)
            for column in cluster_df.columns:
                unified_series[f"cluster_{cluster_id}_{column}"] = cluster_df[column]
        return unified_series

    def get_cluster_transition_matrix(
        self, filepath: str, window_days: int = 7
    ) -> pd.DataFrame:
        """
        Analyze how users transition between clusters over time.

        Parameters:
        -----------
        filepath : str
            Path to CSV file
        window_days : int
            Time window for transition analysis

        Returns:
        --------
        pd.DataFrame
            Cluster transition probabilities
        """
        df = self.base_engineer.load_and_preprocess(filepath)

        # Add cluster IDs
        df_with_clusters = self.clustering_engine.transform(df)

        # Split data into time windows
        df_with_clusters["date"] = df_with_clusters["start_timestamp"].dt.date
        dates = sorted(df_with_clusters["date"].unique())

        if len(dates) <= window_days:
            return pd.DataFrame()  # Not enough data

        transitions = []

        for i in range(len(dates) - window_days):
            # Define time windows
            window_start = dates[i]
            window_end = dates[i + window_days - 1]

            # Get users in first window
            window1_data = df_with_clusters[
                (df_with_clusters["date"] >= window_start)
                & (df_with_clusters["date"] <= window_end)
            ]

            # Get users in next window
            window2_data = df_with_clusters[
                (df_with_clusters["date"] > window_end)
                & (
                    df_with_clusters["date"]
                    <= dates[min(i + 2 * window_days - 1, len(dates) - 1)]
                )
            ]

            # Get user clusters in each window
            user_clusters_window1 = (
                window1_data.groupby("user_id")["cluster_id"]
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
                .dropna()
            )

            user_clusters_window2 = (
                window2_data.groupby("user_id")["cluster_id"]
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
                .dropna()
            )

            # Find common users
            common_users = set(user_clusters_window1.index) & set(
                user_clusters_window2.index
            )

            # Count transitions
            for user in common_users:
                from_cluster = user_clusters_window1[user]
                to_cluster = user_clusters_window2[user]
                transitions.append(
                    {"from_cluster": from_cluster, "to_cluster": to_cluster}
                )

        if not transitions:
            return pd.DataFrame()

        # Create transition matrix
        transitions_df = pd.DataFrame(transitions)
        transition_matrix = pd.crosstab(
            transitions_df["from_cluster"],
            transitions_df["to_cluster"],
            normalize="index",
        ).round(3)

        # Add cluster names
        cluster_names = {
            cid: desc["name"]
            for cid, desc in self.clustering_engine.cluster_descriptions.items()
        }

        transition_matrix.index = [
            f"{i}: {cluster_names.get(i, '')}" for i in transition_matrix.index
        ]
        transition_matrix.columns = [
            f"{i}: {cluster_names.get(i, '')}" for i in transition_matrix.columns
        ]

        return transition_matrix


# Example usage
def main():
    base_engineer = UserBehaviorFeatureEngineer(window_size=60, default_alpha=0.3)
    clustering_engine = UserClusteringEngine(
        duration_percentiles=(33, 66), volume_percentiles=(33, 66)
    )
    enhanced_engineer = EnhancedFeatureEngineer(base_engineer, clustering_engine)
    # Load and fit clustering
    print("Fitting clustering model...")
    df = base_engineer.load_and_preprocess("data.csv")

    clustering_engine.fit(df, method="percentile")

    # Create unified time series
    print("\nCreating unified time series...")
    unified_series = enhanced_engineer.create_cluster_timeseries("data.csv")
    unified_series.to_csv("unified_cluster_series.csv")

    return unified_series


if __name__ == "__main__":
    results = main()
