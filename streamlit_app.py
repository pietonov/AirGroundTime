######################### Import ##############################
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import joblib  # Import joblib for loading models
import os

######################### Load Models #########################
@st.cache_resource
def load_models():
    try:
        glm_path = 'DATA/glm_model.pkl'
        rf_path = 'DATA/rf.pkl'
        if not os.path.exists(glm_path):
            st.error(f"Model file not found: {glm_path}")
            raise FileNotFoundError(f"{glm_path} does not exist.")
        if not os.path.exists(rf_path):
            st.error(f"Model file not found: {rf_path}")
            raise FileNotFoundError(f"{rf_path} does not exist.")
        
        glm_model = joblib.load(glm_path)
        rf_model = joblib.load(rf_path)
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {e}. Ensure the required libraries are installed.")
        raise
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        raise
    return glm_model, rf_model

glm_full, rf = load_models()

######################### Sidebar Navigation ##################
st.sidebar.title("Navigation")

# Initialize a session state for navigation
if "section" not in st.session_state:
    st.session_state["section"] = "Docs"

# Navigation Buttons
if st.sidebar.button("Docs"):
    st.session_state["section"] = "Docs"
if st.sidebar.button("Data Exploratory"):
    st.session_state["section"] = "Data Exploratory"
if st.sidebar.button("Prediction Apps"):
    st.session_state["section"] = "Prediction Apps"

######################### Sections ############################

if st.session_state["section"] == "Docs":
    ######################### Docs ############################
    st.title("Flight Statistics Interactive Documentation")
    st.markdown("""
    ## Documentation

    ### Purpose
    This app visualizes flight statistics in the United States. Users can explore total flights and average ground time based on aircraft configuration type.

    ### Instructions
    - **Select Year:** Use filters to select the year.
    - **Select Aircraft Configuration:** Choose the aircraft configuration.
    - **Interactive Charts:** Hover over data points to see details, zoom in and out as needed.

    ### Data Description
    - **YEAR:** Year of data.
    - **AIRCRAFT_CONFIG_DESC:** Type of aircraft: Passenger, Freight, etc.
    - **total_flights:** Total number of flights.
    - **average_ground_time:** The average ground time.
    """)

elif st.session_state["section"] == "Data Exploratory":
    ######################### Data Exploratory ################
    st.title("Flight Interactive Visualization")
    st.subheader("Flight Statistics")
    
    # Load Data
    @st.cache
    def load_data():
        return pd.read_csv('DATA/summarized_flight_data.csv')
    df = load_data()

    # Filters
    year_filter = st.multiselect(
        "Select Year(s)", options=df['YEAR'].unique(), default=df['YEAR'].unique()
    )
    aircraft_filter = st.multiselect(
        "Select Aircraft Configuration(s)", options=df['AIRCRAFT_CONFIG_DESC'].unique(), default=df['AIRCRAFT_CONFIG_DESC'].unique()
    )
    filtered_df = df[(df['YEAR'].isin(year_filter)) & (df['AIRCRAFT_CONFIG_DESC'].isin(aircraft_filter))]
    st.dataframe(filtered_df)

    # Line Chart
    st.subheader("Total Flights Over Time by Aircraft Configuration")
    fig_line = px.line(
        filtered_df, x='YEAR', y='total_flights', color='AIRCRAFT_CONFIG_DESC',
        title='Total Flights Over Time by Aircraft Configuration'
    )
    st.plotly_chart(fig_line)

    # Bar Chart
    st.subheader("Average Ground Time by Aircraft Configuration")
    bar_chart_data = filtered_df.groupby('AIRCRAFT_CONFIG_DESC').agg(
        average_ground_time=('average_ground_time', 'mean')
    ).reset_index()
    fig_bar = px.bar(
        bar_chart_data, x='AIRCRAFT_CONFIG_DESC', y='average_ground_time',
        title='Average Ground Time by Aircraft Configuration'
    )
    st.plotly_chart(fig_bar)

elif st.session_state["section"] == "Prediction Apps":
    ######################### Prediction Apps #################
    st.title("Predict Ground Time")

    # User Input Form
    st.header("Input Flight Details")
    distance = st.number_input("Distance (miles):", min_value=0, value=500)
    large_airport = st.selectbox("Large Airport:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    has_passengers = st.selectbox("Has Passengers:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    passengers = st.number_input("Number of Passengers:", min_value=0, value=150) if has_passengers == 1 else 0
    is_winter = st.selectbox("Winter Season:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    unique_carrier = st.selectbox(
        "Unique Carrier:",
        options=['American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.', 
                 'Southwest Airlines Co.', 'Alaska Airlines Inc.', 'Other']
    )
    
    # Prepare Data for Prediction
    input_data = {
        'DISTANCE': [distance],
        'LARGE_AIRPORT': [large_airport],
        'HAS_PASSENGERS': [has_passengers],
        'PASSENGERS': [passengers],
        'IS_WINTER': [is_winter],
        'UNIQUE_CARRIER': [unique_carrier]
    }
    input_df = pd.DataFrame(input_data)

    # GLM Prediction
    st.subheader("Predictions")
    if st.button("Predict"):
        try:
            glm_pred = glm_full.predict(input_df)
            st.success(f"GLM Predicted Ground Time: {glm_pred.iloc[0]:.2f} minutes")
        except Exception as e:
            st.error(f"Error during GLM prediction: {e}")

        try:
            rf_input_df = pd.get_dummies(input_df, columns=['UNIQUE_CARRIER'])
            rf_pred = rf.predict(rf_input_df)
            st.success(f"Random Forest Predicted Ground Time: {rf_pred[0]:.2f} minutes")
        except Exception as e:
            st.error(f"Error during Random Forest prediction: {e}")
