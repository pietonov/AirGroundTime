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


st.set_page_config(page_title="Ground Time Predictor", layout="wide")


st.title("Ground Time Predictor 🛬")
st.image("DATA/GroundTimePredictor.png", caption="Ground Operations Illustration by ChatGPT", width=800)

st.write("""
Welcome to the **Ground Time Predictor**, a tool designed to estimate the duration of ground operations for flights based on input parameters such as airport, aircraft type, and operational conditions.
""")

st.subheader("Why Use This App?")
st.write("""
- Optimize flight schedules.
- Improve airport ground efficiency.
- Reduce delays for better customer satisfaction.
""")

st.subheader("How to Use This App")
st.write("""
1. Go to the **Prediction Apps** page from the sidebar.
2. Input details about the flight and ground conditions.
3. Click **Predict** to calculate estimated ground time.
4. Explore insights to improve operational planning.
""")

st.info("Start by navigating to the **Prediction Apps** page in the sidebar!")




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
st.sidebar.title("Navigation")
