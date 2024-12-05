######################### Import ##############################
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import joblib
import os


st.set_page_config(page_title="Ground Time Predictor")

# Header Section
st.title("Ground Time Predictor")
st.image(
    "DATA/GroundTimePredictor.png", 
    caption="Streamlining Ground Operations for Better Efficiency", 
    width=800
)

# Introduction
st.write("""
Welcome to the **Ground Time Predictor**, an advanced tool designed to estimate ground operation times for flights. Whether you're an airline operator, airport planner, or aviation enthusiast, this app provides actionable insights to optimize your operations.
""")


# Key Benefits
st.subheader("Why Choose the Ground Time Predictor?")
st.markdown("""
- **Enhance Efficiency**: Streamline ground operations and reduce turnaround time.
- **Predict Delays**: Get reliable predictions for proactive decision-making.
- **Data-Driven Insights**: Use machine learning models to improve planning and logistics.
- **Scalable Solution**: Suitable for small, medium, and large airports worldwide.
""")


# How to Use Section
st.subheader("How to Use This App")
st.markdown("""
1. Navigate to the **Prediction Apps** page from the sidebar.
2. Enter flight details, including **aircraft type**, **distance**, and other factors.
3. Click the **Predict** button to get the estimated ground time.
4. Explore **visualizations** and **insights** to optimize your planning.
""")

# Feature Highlights
st.subheader("Features")
st.markdown("""
- **Interactive Visualizations**: Dynamic charts to analyze flight and ground statistics.
- **Machine Learning Models**: Leveraging advanced GLM and Random Forest techniques for accurate predictions.
- **Customizable Inputs**: Tailored predictions based on your unique flight parameters.
- **Documentation**: Comprehensive guidance for aviation data science enthusiast.
""")

# Call to Action
st.info("Ready to begin? Head over to the **Prediction Apps** page using the sidebar to start your journey.")
st.info("Need support? Contact us for assistance or inquiries.")






######################### Load Models #########################
@st.cache_resource
def load_models():
    try:
        glm_path = 'DATA/glm_model.pkl'
        rf_path = 'DATA/rf.pkl'
        if not os.path.exists(glm_path) or not os.path.exists(rf_path):
            raise FileNotFoundError("Model files not found.")
        glm_model = joblib.load(glm_path)
        rf_model = joblib.load(rf_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise
    return glm_model, rf_model

glm_full, rf = load_models()

######################### Load Data ###########################
@st.cache
def load_data():
    return pd.read_csv('DATA/summarized_flight_data.csv')

@st.cache
def load_correlation_matrix():
    return pd.read_csv('DATA/correlation_matrix.csv')

@st.cache
def load_histogram_summary():
    return pd.read_csv('DATA/histogram_summary_ground_time.csv')

@st.cache
def load_boxplot_data():
    return pd.read_csv('DATA/boxplot_summary.csv')

@st.cache
def load_qqplot_data():
    return pd.read_csv('DATA/qq_sample.csv')

######################### Sidebar Navigation ##################
# Credits
st.sidebar.title("Credits")