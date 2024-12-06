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
- **Website**: [www.groundtimepredictor.com](https://www.groundtimepredictor.com)  
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

