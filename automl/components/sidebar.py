import streamlit as st


class Sidebar:
    @staticmethod
    def render():
        """Render the sidebar configuration"""
        st.sidebar.header("Configuration")

        # Dataset selection
        dataset_name = st.sidebar.selectbox(
            "Choose Dataset", ["Air Passengers", "Stock Price Example", "Custom"]
        )

        return dataset_name
