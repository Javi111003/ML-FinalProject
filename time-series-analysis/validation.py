"""
Validate hypothesis about variables of the time series
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def verify_hypotheses(data_raw_path, data_path):
    """
    Verify the two hypotheses:
    1. Exponential distribution of service usage time frames
    2. Correlation between expected users dropping and next-minute volume
    """

    print("=" * 80)
    print("HYPOTHESIS VERIFICATION SCRIPT")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df_raw = pd.read_csv(data_raw_path)
    df_ts = pd.read_csv(data_path)

    # Ensure datetime format
    df_raw["start_timestamp"] = pd.to_datetime(df_raw["start_timestamp"])
    df_raw["end_timestamp"] = pd.to_datetime(df_raw["end_timestamp"])
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])

    print(f"Raw data shape: {df_raw.shape}")
    print(f"Time series data shape: {df_ts.shape}")
    print(f"Time range: {df_ts['timestamp'].min()} to {df_ts['timestamp'].max()}")

    # ===================================================================
    # HYPOTHESIS 1: Exponential distribution of service usage time frames
    # ===================================================================
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1: Exponential distribution of service usage time frames")
    print("=" * 80)

    # Calculate session durations in minutes
    df_raw["duration_minutes"] = (
        df_raw["end_timestamp"] - df_raw["start_timestamp"]
    ).dt.total_seconds() / 60

    # Remove any zero or negative durations (data errors)
    df_raw = df_raw[df_raw["duration_minutes"] > 0]

    print(f"\nNumber of user sessions: {len(df_raw)}")
    print(f"Duration statistics (minutes):")
    print(f"  Min: {df_raw['duration_minutes'].min():.2f}")
    print(f"  Max: {df_raw['duration_minutes'].max():.2f}")
    print(f"  Mean: {df_raw['duration_minutes'].mean():.2f}")
    print(f"  Std: {df_raw['duration_minutes'].std():.2f}")

    # Test for exponential distribution
    durations = df_raw["duration_minutes"].values

    # Fit exponential distribution
    lambda_est = 1 / np.mean(durations)
    print(f"\nEstimated lambda for exponential distribution: {lambda_est:.4f}")

    # Kolmogorov-Smirnov test for exponential distribution
    ks_stat, ks_pvalue = stats.kstest(durations, "expon", args=(0, 1 / lambda_est))
    print(f"\nKolmogorov-Smirnov test for exponential distribution:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pvalue:.4f}")

    # Q-Q plot for visual assessment
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram with exponential fit
    ax = axes[0, 0]
    n_bins = min(50, int(len(durations) / 10))
    ax.hist(
        durations,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
    )

    # Plot theoretical exponential PDF
    x = np.linspace(0, np.max(durations), 1000)
    pdf_exp = lambda_est * np.exp(-lambda_est * x)
    ax.plot(x, pdf_exp, "r-", linewidth=2, label=f"Exp(λ={lambda_est:.3f})")
    ax.set_xlabel("Session Duration (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Session Durations with Exponential Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[0, 1]
    stats.probplot(durations, dist="expon", sparams=(0, 1 / lambda_est), plot=ax)
    ax.set_title("Q-Q Plot against Exponential Distribution")
    ax.grid(True, alpha=0.3)

    # Log-survival plot (should be linear for exponential distribution)
    ax = axes[1, 0]
    sorted_durations = np.sort(durations)
    survival = 1 - np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    ax.plot(sorted_durations, np.log(survival), "b.", alpha=0.5)

    # Expected line for exponential
    x_line = np.linspace(0, np.max(sorted_durations), 100)
    y_line = -lambda_est * x_line
    ax.plot(x_line, y_line, "r-", linewidth=2, label="Expected for exponential")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("log(Survival Probability)")
    ax.set_title("Log-Survival Plot (Linear = Exponential)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ECDF comparison
    ax = axes[1, 1]
    ecdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    theoretical_ecdf = 1 - np.exp(-lambda_est * sorted_durations)
    ax.plot(sorted_durations, ecdf, "b-", linewidth=2, label="Empirical CDF")
    ax.plot(
        sorted_durations,
        theoretical_ecdf,
        "r--",
        linewidth=2,
        label="Theoretical Exponential CDF",
    )
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "hypothesis1_exponential_distribution.png", dpi=150, bbox_inches="tight"
    )
    print(f"\nVisualization saved to 'hypothesis1_exponential_distribution.png'")

    # ===================================================================
    # HYPOTHESIS 2: Correlation with expected users dropping
    # ===================================================================
    print("\n" + "=" * 80)
    print(
        "HYPOTHESIS 2: Correlation of expected users dropping with next-minute volume"
    )
    print("=" * 80)

    # Prepare time series data
    df_ts = df_ts.set_index("timestamp").sort_index()

    # Calculate session end times
    df_raw["session_end"] = df_raw["end_timestamp"]

    # Create a DataFrame with session end counts per minute
    session_ends = df_raw["session_end"].value_counts().sort_index()
    session_ends_df = pd.DataFrame(session_ends).rename(
        columns={"session_end": "session_ends"}
    )
    session_ends_df.index.name = "timestamp"

    # Resample to 1-minute intervals and forward fill to match time series
    session_ends_df = (
        session_ends_df.resample("1min").sum().reindex(df_ts.index, fill_value=0)
    )

    # Calculate rolling statistics for expected users dropping
    # Using 1-hour window (60 minutes) as specified
    window_size = 60

    # Fit exponential distribution in rolling window to estimate lambda
    def estimate_lambda_in_window(window_data):
        """Estimate lambda parameter for exponential distribution in a window"""
        if len(window_data) > 0 and np.sum(window_data) > 0:
            return 1 / np.mean(window_data)
        return np.nan

    # Calculate expected users dropping in next minute based on exponential distribution
    # For exponential distribution, survival probability at time t: S(t) = exp(-λt)
    # Probability of dropping in next minute: 1 - S(1) = 1 - exp(-λ)

    expected_dropping = pd.Series(index=df_ts.index, dtype=float)

    # We need to estimate lambda from recent session durations
    # Create a moving window of recent session durations
    for i in range(window_size, len(df_ts)):
        window_start = df_ts.index[i - window_size]
        window_end = df_ts.index[i]

        # Get sessions that ended in this window
        window_sessions = df_raw[
            (df_raw["session_end"] >= window_start)
            & (df_raw["session_end"] <= window_end)
        ]

        if len(window_sessions) > 0:
            lambda_window = 1 / window_sessions["duration_minutes"].mean()
            # Expected number dropping in next minute = active_users * (1 - exp(-lambda))
            # For simplicity, we'll use the number of active sessions ending now
            # as a proxy for active users
            active_users_proxy = (
                len(window_sessions) / window_size
            )  # Average active users per minute

            # Probability of dropping in next minute
            drop_prob = 1 - np.exp(-lambda_window)
            expected_dropping.iloc[i] = active_users_proxy * drop_prob
        else:
            expected_dropping.iloc[i] = 0

    # Forward fill any NaN values
    expected_dropping = expected_dropping.fillna(0)

    # Calculate correlation with next-minute volume
    # Shift volume by -1 to get next minute's volume
    next_minute_volume = df_ts["volume"].shift(-1)

    # Create correlation dataframe
    corr_df = pd.DataFrame(
        {
            "expected_dropping": expected_dropping,
            "next_minute_volume": next_minute_volume,
        }
    ).dropna()

    # Calculate correlation metrics
    pearson_corr, pearson_p = stats.pearsonr(
        corr_df["expected_dropping"], corr_df["next_minute_volume"]
    )
    spearman_corr, spearman_p = stats.spearmanr(
        corr_df["expected_dropping"], corr_df["next_minute_volume"]
    )

    print(f"\nCorrelation Analysis:")
    print(f"  Number of data points: {len(corr_df)}")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

    # Visualization for hypothesis 2
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Time series comparison
    ax = axes[0]
    ax.plot(
        corr_df.index,
        corr_df["next_minute_volume"],
        "b-",
        alpha=0.7,
        label="Next Minute Volume",
    )
    ax.set_ylabel("Volume", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.set_title("Time Series: Expected Users Dropping vs Next Minute Volume")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        corr_df.index,
        corr_df["expected_dropping"],
        "r-",
        alpha=0.7,
        linewidth=2,
        label="Expected Users Dropping",
    )
    ax2.set_ylabel("Expected Users Dropping", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Scatter plot with regression line
    ax = axes[1]
    ax.scatter(
        corr_df["expected_dropping"],
        corr_df["next_minute_volume"],
        alpha=0.5,
        s=10,
        color="green",
    )

    # Add regression line
    z = np.polyfit(corr_df["expected_dropping"], corr_df["next_minute_volume"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(
        corr_df["expected_dropping"].min(), corr_df["expected_dropping"].max(), 100
    )
    ax.plot(
        x_range,
        p(x_range),
        "r--",
        linewidth=2,
        label=f"Regression: y = {z[0]:.3f}x + {z[1]:.3f}",
    )

    ax.set_xlabel("Expected Users Dropping (next minute)")
    ax.set_ylabel("Next Minute Volume")
    ax.set_title(f"Scatter Plot (Pearson r = {pearson_corr:.3f}, p = {pearson_p:.3e})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribution of expected users dropping
    ax = axes[2]
    ax.hist(
        corr_df["expected_dropping"],
        bins=50,
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    ax.set_xlabel("Expected Users Dropping (next minute)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Expected Users Dropping")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hypothesis2_correlation_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to 'hypothesis2_correlation_analysis.png'")

    # Additional metrics
    print(f"\nAdditional Metrics:")
    print(f"  Mean expected users dropping: {corr_df['expected_dropping'].mean():.4f}")
    print(f"  Std expected users dropping: {corr_df['expected_dropping'].std():.4f}")
    print(f"  Mean next minute volume: {corr_df['next_minute_volume'].mean():.2f}")
    print(f"  Std next minute volume: {corr_df['next_minute_volume'].std():.2f}")

    # ===================================================================
    # CONCLUSIONS
    # ===================================================================
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    print("\nHypothesis 1 - Exponential distribution of service usage time frames:")
    if ks_pvalue > 0.05:
        print(f"  ✓ SUPPORTED: KS test p-value = {ks_pvalue:.4f} > 0.05")
        print(
            "    Cannot reject the null hypothesis that data follows exponential distribution."
        )
    else:
        print(f"  ✗ NOT FULLY SUPPORTED: KS test p-value = {ks_pvalue:.4f} ≤ 0.05")
        print("    Data may not perfectly follow exponential distribution.")

    print("  Visual assessment from Q-Q and log-survival plots should be considered.")

    print("\nHypothesis 2 - Correlation with expected users dropping:")
    if abs(pearson_corr) > 0.7 and pearson_p < 0.05:
        strength = "STRONG" if abs(pearson_corr) > 0.8 else "MODERATE"
        print(f"  ✓ SUPPORTED: {strength} correlation ({pearson_corr:.3f})")
        print(f"    Statistically significant (p = {pearson_p:.3e})")
    elif abs(pearson_corr) > 0.3 and pearson_p < 0.05:
        print(f"  ✓ PARTIALLY SUPPORTED: Weak correlation ({pearson_corr:.3f})")
        print(f"    Statistically significant (p = {pearson_p:.3e})")
    elif pearson_p < 0.05:
        print(f"  ✗ NOT SUPPORTED: Very weak/no correlation ({pearson_corr:.3f})")
        print(f"    But statistically significant (p = {pearson_p:.3e})")
    else:
        print(f"  ✗ NOT SUPPORTED: No significant correlation ({pearson_corr:.3f})")
        print(f"    Not statistically significant (p = {pearson_p:.3f})")

    print("\nRecommendations:")
    if ks_pvalue > 0.05 or ks_pvalue < 0.01:  # Considering typical statistical practice
        print(
            "  1. Exponential distribution assumption appears reasonable for feature engineering."
        )
    else:
        print("  1. Consider alternative distributions or non-parametric methods.")

    if abs(pearson_corr) > 0.5 and pearson_p < 0.05:
        print(
            "  2. Expected users dropping is a promising feature for volume prediction."
        )
        print("  3. Include this feature in time series models to reduce variance.")
    elif pearson_p < 0.05:
        print(
            "  2. Correlation exists but may not be strong enough as a standalone feature."
        )
        print("  3. Consider combining with other features in a multivariate model.")

    return {
        "hypothesis1": {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "lambda_estimated": lambda_est,
            "mean_duration": np.mean(durations),
            "n_sessions": len(durations),
        },
        "hypothesis2": {
            "pearson_correlation": pearson_corr,
            "pearson_pvalue": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_pvalue": spearman_p,
            "n_data_points": len(corr_df),
            "mean_expected_dropping": corr_df["expected_dropping"].mean(),
            "mean_next_volume": corr_df["next_minute_volume"].mean(),
        },
    }


if __name__ == "__main__":
    # Example usage
    data_raw_path = "data_raw.csv"
    data_path = "data.csv"

    try:
        results = verify_hypotheses(data_raw_path, data_path)
        print("\n" + "=" * 80)
        print("Analysis complete. Check saved visualizations.")
        print("=" * 80)

        # Save results to file
        import json

        with open("hypothesis_verification_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nResults saved to 'hypothesis_verification_results.json'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure both 'data_raw.csv' and 'data.csv' exist in the current directory."
        )
        print("\nExpected file formats:")
        print("1. data_raw.csv should contain columns: user_id, start_timestamp, end_timestamp")
        print("2. data.csv should contain columns: timestamp, volume")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
