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
st.subheader("Ground Time Predictor Overview")
st.write("""
Welcome to the **Ground Time Predictor**, an advanced tool designed to estimate ground operation times for flights. This application combines cutting-edge machine learning models with robust data analysis techniques to streamline ground operation planning. Whether you're an airline operator, airport planner, or aviation enthusiast, this app provides actionable insights to optimize your operations.

Designed for aviation professionals and researchers, the Ground Time Predictor is a comprehensive tool for analyzing trends, improving efficiency, and making data-driven decisions to enhance the overall ground handling process.
""")


# Key Benefits
st.subheader("Application Highlights")
st.markdown("""
- **Enhance Efficiency**: Streamline ground operations and reduce turnaround time with advanced prediction models.
- **Predict Delays**: Get reliable predictions using Generalized Linear Models (GLM) and Random Forest for proactive decision-making.
- **Data-Driven Insights**: Leverage feature importance rankings, correlation analysis, and trend visualizations to improve planning and logistics.
- **Scalable Solution**: Designed for airports of all sizes, from regional hubs to international gateways, supporting global applicability.
- **Comprehensive Documentation**: Ensure transparency and reproducibility with detailed guidance for aviation professionals and researchers.
""")


# How to Use Section
st.subheader("How to Use This App")
st.markdown("""
1. Navigate to the **Prediction Apps** page from the sidebar.
2. Enter flight details, including **aircraft type**, **distance**, and other factors.
3. Click the **Predict** button to get the estimated ground time.
4. Explore **visualizations** and **insights** to optimize your planning.
""")


# Insights Section
st.subheader("Insights at a Glance")
st.markdown("""
- **Ground Time Drivers**:  
  - Longer distances reduce ground time marginally.  
  - Winter conditions significantly increase delays.  
  - Passenger-heavy flights require more turnaround time.
- **Efficiency Variability**: Airlines with dedicated ground handling teams consistently outperform their peers.  
""")

# Assumptions and Methodology
st.subheader("Assumptions & Methodology")
st.markdown("""
- **Generalized Linear Models (GLM)**: Assumes a Gaussian distribution of ground time and linear relationships with predictors.  
- **Random Forest**: Handles complex interactions and non-linear relationships with high accuracy, ensuring robustness.  
- **Feature Selection**: Incorporates variables based on domain expertise and rigorous statistical testing to maintain relevance and significance.  
- **Data Integrity**: Missing data was addressed using median imputation to ensure consistency and reliability in model performance.  
""")


# Call to Action
st.info("Ready to begin? Head over to the **Prediction Apps** page using the sidebar to start your journey.")
st.info("Need support? Contact us for assistance or inquiries.")


# Acknowledgments
st.subheader("Acknowledgments")
st.markdown("""
This project was made possible with the following resources and references:  
- **Bureau of Transportation Statistics (BTS)**: For providing comprehensive aviation datasets that informed this analysis.  
- **OurAirports Data**: For offering detailed airport-specific metrics used in the modeling process.  
- **Society of Actuaries (SOA)**: For their resources on statistical modeling techniques, which guided the methodology.  

These organizations publicly available data and materials were critical in development.
""")


######################### Sidebar Navigation ##################


# Key Metrics
st.sidebar.title("Key Metrics")
st.sidebar.markdown("""
- **Training Period**: 2014 - 2022  
- **Testing Period**: 2023  
- **Total Flights Analyzed**: 137,522  
- **Top Airlines by Ground Efficiency**:  
  - Southwest Airlines Co.  
  - Spirit Airlines  
  - Delta Air Lines Inc.  
""")



# Contact Information
st.sidebar.title("Contact & Support")
st.sidebar.markdown("""
For professional inquiries or feedback:  
- **Email**: support@groundtimepredictor.com  
- **Phone**: +1 (555) 987-6543
""")





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
