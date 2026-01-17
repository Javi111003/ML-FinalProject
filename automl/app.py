import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from config import (
    DEFAULT_MODEL_CONFIGS,
    FEATURE_ENGINEERING_CONFIG,
    PAGE_CONFIG,
    AUTOML_DEFAULTS,
)
from models.factory import ModelFactory
from components.explorer import DataExplorer
from components.comparison import ModelComparison
from components.sidebar import Sidebar
from components.pipeline import AutoMLPipeline
from utils.helpers import setup_warnings

# Setup
setup_warnings()
st.set_page_config(**PAGE_CONFIG)


class TimeSeriesExplorerApp:
    """Main application class with unified model interface"""

    def __init__(self):
        self.data_explorer = DataExplorer()
        self.model_registry = DEFAULT_MODEL_CONFIGS.copy()
        self.automl_pipeline = None
        self.best_models = []

    def run(self):
        """Run the main application"""
        st.title("🤖 AutoML Time Series Forecasting")
        st.markdown(
            """
        This tool automatically selects the best time series forecasting model 
        using hyperparameter optimization and feature engineering.
        """
        )

        # Sidebar configuration
        dataset_name = Sidebar.render()

        # Custom model management in sidebar
        self._render_custom_models_section()

        if not self.data_explorer.load_data(dataset_name):
            return

        # Main tabs
        tab1, tab2 = st.tabs(["🤖 AutoML Pipeline", "⚙️ Custom Models"])

        with tab1:
            self._render_automl_pipeline()

        with tab2:
            self._render_custom_models_interface()

    def _render_custom_models_section(self):
        """Render custom model management section in sidebar"""
        with st.sidebar.expander("Custom Models", expanded=False):
            st.subheader("Manage Custom Models")

            # Add custom model
            with st.form("add_custom_model"):
                st.write("Add a custom model")
                custom_name = st.text_input("Model Name")
                custom_class = st.text_input(
                    "Class Path (e.g., sklearn.ensemble.RandomForestRegressor)"
                )
                custom_type = st.selectbox("Model Type", ["statistical", "ml"])

                # Simple parameter input (could be enhanced)
                custom_params = st.text_area(
                    "Parameters (JSON format)",
                    value='{"param1": [value1, value2]}',
                    help="Enter as JSON with parameter names and lists of values",
                )

                if st.form_submit_button("Add Custom Model"):
                    try:
                        import json

                        params_dict = json.loads(custom_params)

                        custom_config = {
                            "class": custom_class,
                            "type": custom_type,
                            "params": params_dict,
                        }

                        ModelFactory.validate_model_config(custom_config)
                        self.model_registry[custom_name] = custom_config
                        st.success(f"Added custom model: {custom_name}")
                    except Exception as e:
                        st.error(f"Error adding custom model: {str(e)}")

            # Remove custom model
            custom_models = [
                m for m in self.model_registry.keys() if m not in DEFAULT_MODEL_CONFIGS
            ]

            if custom_models:
                st.subheader("Remove Custom Models")
                model_to_remove = st.selectbox("Select model to remove", custom_models)
                if st.button("Remove Selected Model"):
                    del self.model_registry[model_to_remove]
                    st.success(f"Removed model: {model_to_remove}")

    def _render_automl_pipeline(self):
        """Render the AutoML pipeline interface"""
        st.header("🔬 AutoML Model Selection")

        # AutoML configuration
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            strategy = st.selectbox(
                "Search Strategy",
                ["optuna_bayesian", "grid_search", "random_search"],
                help="Bayesian optimization is recommended for efficiency",
                index=0,
            )

        with col2:
            metric = st.selectbox(
                "Optimization Metric",
                ["rmse", "mae", "r2"],
                help="Metric to optimize during model selection",
                index=0,
            )

        with col3:
            n_trials = st.slider(
                "Trials per Model",
                min_value=1,
                max_value=100,
                value=5,
                help="Number of hyperparameter combinations to try",
            )
        with col4:
            forecast_steps = st.slider(
                "Forecast Steps", 1, 365, 30, key="comparison_forecast_steps"
            )
            self.forecast_steps = forecast_steps
        with col5:
            freq = st.text_input("Frequency (e.g. 'D', 'h', 'min')", "1min")

        # Model selection
        st.subheader("Select Models to Evaluate")
        available_models = ModelFactory.get_available_models(self.model_registry)
        selected_models = st.multiselect(
            "Choose models for AutoML",
            available_models,
            default=[
                "ARIMA",
                "Random Forest",
            ],
        )

        # Feature engineering options
        st.subheader("Feature Engineering Options")
        feature_options = self._render_feature_engineering_options()

        if st.button("🚀 Run AutoML Pipeline", type="primary"):
            if not selected_models:
                st.error("Please select at least one model to evaluate")
                return
            self.data_explorer.df.index.freq = freq

            # Initialize AutoML pipeline
            self.automl_pipeline = AutoMLPipeline(
                model_registry=self.model_registry,
                feature_config=FEATURE_ENGINEERING_CONFIG,
                strategy=strategy,
                freq=freq,
                metric=metric,
            )

            # Run AutoML
            with st.spinner("Running AutoML pipeline. This may take a few minutes..."):
                self.best_models = self.automl_pipeline.run(
                    series=self.data_explorer.df,
                    dates=self.data_explorer.df.index,
                    selected_models=selected_models,
                    feature_options=feature_options,
                    n_trials_per_model=n_trials,
                )

            # Display results
            ModelComparison.display_results(self.best_models, metric)

            # Detailed comparison if we have models
            if len(self.best_models) >= 2:
                self._render_detailed_comparison()

    def _render_feature_engineering_options(self) -> Dict[str, Any]:
        """Render feature engineering options"""
        col1, col2, col3 = st.columns(3)

        feature_options = {}

        with col1:
            use_lags = st.checkbox("Lag Features", value=True)
            if use_lags:
                max_lags = st.slider("Max Lags", 1, 5, 3)
                feature_options["lag_features"] = True
                feature_options["max_lags"] = max_lags
            else:
                feature_options["lag_features"] = False

        with col2:
            use_rolling = st.checkbox("Rolling Statistics", value=False)
            feature_options["rolling_features"] = use_rolling

        with col3:
            use_seasonal = st.checkbox("Seasonal Features", value=True)
            feature_options["seasonal_features"] = use_seasonal

        return feature_options

    def _render_detailed_comparison(self):
        """Render detailed model comparison section"""
        st.subheader("🔍 Detailed Model Comparison")

        if len(self.best_models) >= 2:
            model_options = [
                f"{i+1}. {m.model_name} ({m.score:.4f})"
                for i, m in enumerate(self.best_models[:5])
            ]

            col1, col2 = st.columns(2)
            with col1:
                model1_idx = st.selectbox(
                    "Select first model for comparison",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x],
                    key="model1_select",
                )

            with col2:
                model2_idx = st.selectbox(
                    "Select second model for comparison",
                    range(len(model_options)),
                    index=1,
                    format_func=lambda x: model_options[x],
                    key="model2_select",
                )

            if model1_idx != model2_idx:
                ModelComparison.compare_selected_models(
                    model1=self.best_models[model1_idx],
                    model2=self.best_models[model2_idx],
                    series=self.data_explorer.df,
                    automl_pipeline=self.automl_pipeline,
                    forecast_steps=self.forecast_steps,
                )
        else:
            st.info("Need at least 2 models for comparison")

    def _render_custom_models_interface(self):
        """Render interface for managing custom models"""
        st.header("⚙️ Custom Model Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Available Models")

            # Show all available models
            model_categories = {
                "Statistical Models": [],
                "ML Models": [],
                "Custom Models": [],
            }

            for name, config in self.model_registry.items():
                if name in DEFAULT_MODEL_CONFIGS:
                    if config["type"] == "statistical":
                        model_categories["Statistical Models"].append(name)
                    else:
                        model_categories["ML Models"].append(name)
                else:
                    model_categories["Custom Models"].append(name)

            for category, models in model_categories.items():
                if models:
                    with st.expander(
                        f"{category} ({len(models)})",
                        expanded=category == "Custom Models",
                    ):
                        for model in sorted(models):
                            config = self.model_registry[model]
                            st.write(f"**{model}**")
                            st.caption(f"Type: {config['type']}")
                            st.caption(f"Class: {config['class']}")

        with col2:
            st.subheader("Add New Model")

            with st.form("new_custom_model"):
                model_name = st.text_input("Model Name")

                col_a, col_b = st.columns(2)
                with col_a:
                    model_type = st.selectbox("Model Type", ["statistical", "ml"])
                with col_b:
                    model_library = st.selectbox(
                        "Library", ["scikit-learn", "statsmodels", "other"]
                    )

                model_class = st.text_input(
                    "Full Class Path",
                    help="e.g., sklearn.ensemble.RandomForestRegressor or statsmodels.tsa.arima.model.ARIMA",
                )

                # Parameter configuration
                st.subheader("Hyperparameter Search Space")
                st.write("Define the parameter grid for hyperparameter optimization")

                param_name = st.text_input("Parameter Name")
                param_values = st.text_area(
                    "Parameter Values (comma-separated)",
                    help="Enter values as: value1, value2, value3",
                )

                params_dict = {}
                if param_name and param_values:
                    values_list = [v.strip() for v in param_values.split(",")]
                    # Try to convert to appropriate types
                    typed_values = []
                    for v in values_list:
                        try:
                            if v.lower() == "true":
                                typed_values.append(True)
                            elif v.lower() == "false":
                                typed_values.append(False)
                            elif v.lower() == "none":
                                typed_values.append(None)
                            elif "." in v:
                                typed_values.append(float(v))
                            else:
                                typed_values.append(int(v))
                        except:
                            typed_values.append(v)

                    params_dict = {param_name: typed_values}

                if st.form_submit_button("Add Model"):
                    if model_name and model_class:
                        try:
                            # Validate the class exists
                            import importlib

                            module_path, class_name = model_class.rsplit(".", 1)
                            module = importlib.import_module(module_path)
                            getattr(module, class_name)

                            # Add to registry
                            self.model_registry[model_name] = {
                                "class": model_class,
                                "type": model_type,
                                "params": params_dict,
                            }

                            st.success(f"Added model: {model_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding model: {str(e)}")
                    else:
                        st.error("Please provide both model name and class path")


def main():
    """Main entry point"""
    app = TimeSeriesExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
