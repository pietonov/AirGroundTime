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


st.title("Ground Time Prediction")

unique_carriers = [
    'American Airlines Inc.',
    'Delta Air Lines Inc.',
    'United Air Lines Inc.',
    'Southwest Airlines Co.',
    'Alaska Airlines Inc.',
    'SkyWest Airlines Inc.',
    'Federal Express Corporation',
    'Spirit Air Lines',
    'Endeavor Air Inc.',
    'ExpressJet Airlines LLC d/b/a aha!',
    'Republic Airline',
    'United Parcel Service',
    'Envoy Air',
    'JetBlue Airways',
    'GoJet Airlines LLC d/b/a United Express',
    'Shuttle America Corp.',
    'PSA Airlines Inc.',
    'Compass Airlines',
    'Frontier Airlines Inc.',
    'Chautauqua Airlines Inc.',
    'Other'
]

# User Input Form
with st.form("prediction_form"):
    st.header("Input Flight Details")
    distance = st.number_input("Distance (miles):", min_value=0, value=500)
    large_airport = st.selectbox("Large Airport:", [1, 0], format_func=lambda x: "Yes" if x else "No")
    has_passengers = st.selectbox("Has Passengers:", [1, 0], format_func=lambda x: "Yes" if x else "No")
    passengers = st.number_input("Number of Passengers:", min_value=0, value=150, disabled=not has_passengers)
    is_winter = st.selectbox("Winter Season:", [1, 0], format_func=lambda x: "Yes" if x else "No")
    unique_carrier = st.selectbox("Carrier:", unique_carriers)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'DISTANCE': [distance],
        'LARGE_AIRPORT': [large_airport],
        'HAS_PASSENGERS': [has_passengers],
        'PASSENGERS': [0 if not has_passengers else passengers],
        'IS_WINTER': [is_winter],
        'UNIQUE_CARRIER': [unique_carrier]
    }
    input_df = pd.DataFrame(input_data)

    try:
        glm_pred = glm_full.predict(input_df)
        st.success(f"GLM Predicted Ground Time: {glm_pred.iloc[0]:.2f} minutes")
    except Exception as e:
        st.error(f"Error during GLM prediction: {e}")

    try:
        rf_input_df = pd.get_dummies(input_df, columns=['UNIQUE_CARRIER'])
        rf_input_df = rf_input_df.reindex(columns=rf.feature_names_in_, fill_value=0)
        rf_pred = rf.predict(rf_input_df)
        st.success(f"Random Forest Predicted Ground Time: {rf_pred[0]:.2f} minutes")
    except Exception as e:
        st.error(f"Error during Random Forest prediction: {e}")


######################### Sidebar Navigation ##################

# Key Statistics
st.sidebar.title("Key Statistics")
st.sidebar.markdown("""
- **Training Data**: 2014 - 2022  
- **Test Data**: 2023  
- **Total Flights Analyzed**: 137,522  
- **Top Airlines by Flight Volume**:  
  - Delta Air Lines Inc.  
  - Southwest Airlines Co.  
  - American Airlines Inc.  
""")

# Assumptions Section
st.sidebar.title("Model Assumptions")
st.sidebar.markdown("""
- **GLM**: Assumes Gaussian distribution for ground time.  
- **Random Forest**: Non-parametric model, optimized for prediction accuracy.  
- Features like `DISTANCE`, `PASSENGERS`, and `IS_WINTER` are key predictors.  
- All categorical variables are one-hot encoded for consistent modeling.
""")

# Performance Highlights
st.sidebar.title("Model Performance")
st.sidebar.markdown("""
- **GLM Weighted RMSE**: 20.0091 minutes  
- **Random Forest Weighted RMSE**: 22.5306 minutes  
- Both models highlight the importance of operational factors like carrier type, airport size, and seasonality.
""")

# Contact Information
st.sidebar.title("Contact Us")
st.sidebar.markdown("""
For support or inquiries:  
- **Email**: support@groundtimepredictor.com  
- **Phone**: +1 (555) 123-4567  
""")

# Credits
st.sidebar.title("Credits")
st.sidebar.markdown("https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FIM")
st.sidebar.markdown("https://ourairports.com/data/")
st.sidebar.markdown("https://www.soa.org/education/exam-req/edu-exam-atpa/")
st.sidebar.markdown("https://chatgpt.com/")