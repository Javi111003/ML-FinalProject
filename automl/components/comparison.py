import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from models.base import ModelResult


class ModelComparison:
    """Component for comparing model results"""

    @staticmethod
    def display_results(best_models: List[ModelResult], metric: str = "rmse"):
        """Display AutoML results"""
        if not best_models:
            st.warning("No models were successfully trained")
            return

        st.header("📊 AutoML Results")

        # Create results table
        results_data = []
        for i, result in enumerate(best_models[:10]):  # Top 10 models
            results_data.append(
                {
                    "Rank": i + 1,
                    "Model": result.model_name,
                    "Type": result.model_type.upper(),
                    "Best Score": f"{result.score:.4f}",
                    "Parameters": (
                        str(result.params)[:100] + "..."
                        if len(str(result.params)) > 100
                        else str(result.params)
                    ),
                }
            )

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df)

        # Plot model comparison
        ModelComparison._plot_model_comparison(best_models, metric)

        # Show top model details
        if best_models:
            ModelComparison._show_top_model_details(best_models[0])

    @staticmethod
    def _plot_model_comparison(best_models: List[ModelResult], metric: str):
        """Plot model performance comparison"""
        st.subheader("Model Performance Comparison")

        top_n = min(5, len(best_models))
        models = [r.model_name for r in best_models[:top_n]]
        scores = [r.score for r in best_models[:top_n]]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, scores, color=colors)

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel(f"{metric.upper()} (lower is better)")
        ax.set_title(f"Top {top_n} Models")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        st.pyplot(fig)

    @staticmethod
    def _show_top_model_details(top_model: ModelResult):
        """Show details of the top-performing model"""
        st.subheader("🏆 Top Model Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Model", top_model.model_name)
            st.metric("Type", top_model.model_type.upper())

        with col2:
            st.metric("Score", f"{top_model.score:.4f}")
            if top_model.features_shape:
                st.metric("Features", f"{top_model.features_shape[1]}")

        # Show parameters in expandable section
        with st.expander("View Model Parameters"):
            st.json(top_model.params)

    @staticmethod
    def compare_selected_models(
        model1: ModelResult,
        model2: ModelResult,
        series: pd.Series,
        automl_pipeline,
        forecast_steps: int = 30,
    ):
        """Compare two selected models in detail"""
        st.header(f"📈 Comparison: {model1.model_name} vs {model2.model_name}")

        # Generate forecasts
        forecast1 = automl_pipeline.forecast(model1, series, forecast_steps)
        forecast2 = automl_pipeline.forecast(model2, series, forecast_steps)

        # Display forecasts side by side
        col1, col2 = st.columns(2)

        for i, (model, forecast, col) in enumerate(
            [(model1, forecast1, col1), (model2, forecast2, col2)]
        ):
            with col:
                st.subheader(model.model_name)

                # Forecast plot
                fig, ax = plt.subplots(figsize=(10, 4))

                # Last 100 historical points
                historical = (
                    series[series.columns[0]].iloc[-100 + forecast_steps :]
                    if len(series) > 100
                    else series
                )
                ax.plot(
                    historical.index,
                    historical.values,
                    label="Historical",
                    color="black",
                    alpha=0.7,
                    linewidth=2,
                )

                # Forecast
                ax.plot(
                    forecast.index[-100:],
                    forecast.values[-100:],
                    label="Forecast",
                    color="red" if i == 0 else "blue",
                    linestyle="--",
                    linewidth=2,
                )

                ax.set_title(f"{model.model_name} {forecast_steps}-step Forecast")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        # Combined comparison
        ModelComparison._plot_combined_forecast(
            series, forecast1, forecast2, model1.model_name, model2.model_name
        )

        # Statistical comparison
        ModelComparison._display_statistical_comparison(
            forecast1, forecast2, model1.model_name, model2.model_name
        )

    @staticmethod
    def _plot_combined_forecast(
        series: pd.Series,
        forecast1: pd.Series,
        forecast2: pd.Series,
        name1: str,
        name2: str,
    ):
        """Plot combined forecast view"""
        st.subheader("Combined Forecast View")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Historical data
        # historical = series.iloc[-100:] if len(series) > 100 else series
        historical = series[series.columns[0]]
        ax.plot(
            historical.index,
            historical.values,
            label="Historical",
            color="black",
            linewidth=2,
            alpha=0.8,
        )

        # Forecasts
        ax.plot(
            forecast1.index,
            forecast1.values,
            label=name1,
            color="red",
            linestyle="--",
            linewidth=2,
        )
        ax.plot(
            forecast2.index,
            forecast2.values,
            label=name2,
            color="blue",
            linestyle="--",
            linewidth=2,
        )

        # Confidence intervals (simplified)
        ax.fill_between(
            forecast1.index,
            forecast1.values * 0.95,
            forecast1.values * 1.05,
            alpha=0.2,
            color="red",
            label=f"{name1} ±5%",
        )
        ax.fill_between(
            forecast2.index,
            forecast2.values * 0.95,
            forecast2.values * 1.05,
            alpha=0.2,
            color="blue",
            label=f"{name2} ±5%",
        )

        ax.set_title("Forecast Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    @staticmethod
    def _display_statistical_comparison(
        forecast1: pd.Series, forecast2: pd.Series, name1: str, name2: str
    ):
        """Display statistical comparison table"""
        st.subheader("Statistical Comparison")

        comparison_data = {
            "Metric": [
                "Mean Forecast",
                "Std Forecast",
                "Min Forecast",
                "Max Forecast",
                "Forecast Range",
                "Trend (Slope)",
            ],
            name1: [
                forecast1.mean(),
                forecast1.std(),
                forecast1.min(),
                forecast1.max(),
                forecast1.max() - forecast1.min(),
                np.polyfit(range(len(forecast1)), forecast1.values, 1)[0],
            ],
            name2: [
                forecast2.mean(),
                forecast2.std(),
                forecast2.min(),
                forecast2.max(),
                forecast2.max() - forecast2.min(),
                np.polyfit(range(len(forecast2)), forecast2.values, 1)[0],
            ],
            "Difference": [
                forecast1.mean() - forecast2.mean(),
                forecast1.std() - forecast2.std(),
                forecast1.min() - forecast2.min(),
                forecast1.max() - forecast2.max(),
                (forecast1.max() - forecast1.min())
                - (forecast2.max() - forecast2.min()),
                np.polyfit(range(len(forecast1)), forecast1.values, 1)[0]
                - np.polyfit(range(len(forecast2)), forecast2.values, 1)[0],
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)

        # Format the display
        def format_value(x):
            if isinstance(x, (int, np.integer)):
                return f"{x:,}"
            elif isinstance(x, float):
                return f"{x:,.4f}"
            return str(x)

        styled_df = comparison_df.style.format(
            {name1: "{:,.4f}", name2: "{:,.4f}", "Difference": "{:+,.4f}"}
        )

        st.dataframe(styled_df)
