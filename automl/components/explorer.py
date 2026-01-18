import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class DataExplorer:
    def __init__(self, freq="min"):
        self.df = None
        self.value_col = None

    def load_data(self, data_name):
        """Load time series dataset by name or from file"""
        if data_name == "Air Passengers":
            df = pd.read_csv("airline-passengers.csv")
            df["Month"] = pd.to_datetime(df["Month"])
            df.set_index("Month", inplace=True)
            self.value_col = "Passengers"
            self.df = df
        elif data_name == "Stock Price Example":
            dates = pd.date_range("2010-01-01", "2020-12-31", freq="D")
            np.random.seed(42)
            prices = np.cumsum(np.random.randn(len(dates)) * 0.01) + 100
            self.df = pd.DataFrame({"Price": prices}, index=dates)
            self.value_col = "Price"
        elif data_name == "Custom":
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if len(df.columns) >= 2:
                    datetime_col = df.columns[0]
                    value_col = df.columns[1]
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                    df[value_col] = df[value_col].fillna(method="ffill")
                    self.value_col = value_col

                    df = df.fillna(0)
                    df = df.replace(np.inf, 2**32)
                    self.df = df
                else:
                    st.error("CSV must have at least 2 columns: datetime and value")
                    return False
            else:
                st.warning("Please upload a CSV file")
                return False
        else:
            st.error("Unknown dataset")
            return False

        return True

    def exploratory_analysis(self):
        """Basic exploratory analysis"""
        st.header("📈 Exploratory Data Analysis")

        if self.df is None:
            return

        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observations", len(self.df))
        with col2:
            st.metric(
                "Time Range",
                f"{self.df.index.min().date()} to {self.df.index.max().date()}",
            )
        with col3:
            st.metric(
                "Frequency",
                str(self.df.index.freq) if self.df.index.freq else "Irregular",
            )

        # Time series plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df.index, self.df[self.value_col], linewidth=2)
        ax.set_title(f"Time Series: {self.value_col}")
        ax.set_ylabel(self.value_col)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Decomposition
        st.subheader("Seasonal Decomposition")
        try:
            decomposition = seasonal_decompose(self.df[self.value_col], period=12)
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            components = [
                decomposition.observed,
                decomposition.trend,
                decomposition.seasonal,
                decomposition.resid,
            ]
            titles = ["Observed", "Trend", "Seasonal", "Residual"]

            for ax, comp, title in zip(axes, components, titles):
                ax.plot(comp)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Decomposition failed: {str(e)}")
