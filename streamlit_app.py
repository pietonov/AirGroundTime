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

# Load the GLM and RF models from the DATA folder
@st.cache_resource  # Cache the models to avoid reloading on every app refresh
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

# Call the load_models function and assign models to variables
glm_full, rf = load_models()

######################### Tabs Navigation ###############################

# Create tabs for navigation
tabs = st.tabs(["Docs", "Data Exploratory", "Prediction Apps"])

######################### Docs Section ###############################
with tabs[0]:
    st.title("Flight Statistics Interactive Documentation")
    
    st.markdown("""
    ## Documentation
    
    ### Purpose
    This app visualizes flight statistics in the United States. Users can explore the total number of flights and the average ground time based on aircraft configuration type.
    
    ### AI Usage
    Code was generated with the assistance of GPT-4, incorporating student input, review, and edits.
    
    ### Instructions
    - **Select Year:** Use the filters to select the year.
    - **Select Aircraft Configuration:** Choose the aircraft configuration.
    - **Interactive Charts:** Hover over data points to see details; zoom in and out as needed.
    - **Heatmap:** Use the dropdown to choose the colormap and the features you want to focus on.
    
    ### Data Description
    - **YEAR:** Year of data.
    - **AIRCRAFT_CONFIG_DESC:** Type of aircraft: Passenger Flight, Freight, Sea, etc.
    - **total_flights:** Total number of flights.
    - **average_ground_time:** The average ground time.
    """)

######################### Data Exploratory Section ###############################
with tabs[1]:
    # Load the data
    def load_data():
        return pd.read_csv('DATA/summarized_flight_data.csv')
    
    def load_correlation_matrix():
        return pd.read_csv('DATA/correlation_matrix.csv')
    
    def load_histogram_summary():
        return pd.read_csv('DATA/histogram_summary_ground_time.csv')
    
    df = load_data()
    full_correlation_matrix_df = load_correlation_matrix()
    histogram_summary = load_histogram_summary()
    
    st.title("Flight Interactive Visualization")
    
    # Filters
    st.subheader("Flight Statistics")
    year_filter = st.multiselect(
        "Select Year(s)", options=df['YEAR'].unique(), default=df['YEAR'].unique()
    )
    aircraft_filter = st.multiselect(
        "Select Aircraft Configuration(s)", options=df['AIRCRAFT_CONFIG_DESC'].unique(), default=df['AIRCRAFT_CONFIG_DESC'].unique()
    )
    
    # Filtered Data
    filtered_df = df[(df['YEAR'].isin(year_filter)) & (df['AIRCRAFT_CONFIG_DESC'].isin(aircraft_filter))]
    
    # Display the filtered dataset
    st.dataframe(filtered_df)
    
    # Line chart: Total Flights Over Time
    st.subheader("Total Flights Over Time by Aircraft Configuration")
    fig_line = px.line(
        filtered_df, 
        x='YEAR', 
        y='total_flights', 
        color='AIRCRAFT_CONFIG_DESC',
        title='Total Flights Over Time by Aircraft Configuration'
    )
    st.plotly_chart(fig_line)
    
    # Bar chart: Average Ground Time
    st.subheader("Average Ground Time by Aircraft Configuration")
    bar_chart_data = filtered_df.groupby('AIRCRAFT_CONFIG_DESC').agg(
        average_ground_time=('average_ground_time', 'mean')
    ).reset_index()
    fig_bar = px.bar(
        bar_chart_data, 
        x='AIRCRAFT_CONFIG_DESC', 
        y='average_ground_time', 
        title='Average Ground Time by Aircraft Configuration'
    )
    st.plotly_chart(fig_bar)

######################### Prediction Apps Section ###############################
with tabs[2]:
    st.title("Predict Ground Time")

    # User Input Form
    st.header("Input Flight Details")
    distance = st.number_input("Distance (miles):", min_value=0, value=500)
    large_airport = st.selectbox("Large Airport:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    has_passengers = st.selectbox("Has Passengers:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    
    # Conditionally display the 'Number of Passengers' input
    passengers = 0
    if has_passengers == 1:
        passengers = st.number_input("Number of Passengers:", min_value=1, value=150)
    else:
        st.write("Number of Passengers: 0 (No passengers)")
    
    is_winter = st.selectbox("Winter Season:", [1, 0], format_func=lambda x: 'Yes' if x else 'No')
    unique_carrier = st.selectbox(
        "Unique Carrier:", 
        ['American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.', 
         'Southwest Airlines Co.', 'Alaska Airlines Inc.', 'Other']
    )
    
    # Prepare input data
    input_data = {
        'DISTANCE': [distance],
        'LARGE_AIRPORT': [large_airport],
        'HAS_PASSENGERS': [has_passengers],
        'PASSENGERS': [passengers],
        'IS_WINTER': [is_winter],
        'UNIQUE_CARRIER': [unique_carrier]
    }
    input_df = pd.DataFrame(input_data)
    glm_input_df = input_df.copy()
    rf_input_df = pd.get_dummies(input_df, columns=['UNIQUE_CARRIER']).reindex(columns=rf.feature_names_in_, fill_value=0)

    # Prediction Button
    if st.button("Predict Ground Time"):
        try:
            glm_pred = glm_full.predict(glm_input_df)
            st.success(f"**GLM Predicted Ground Time:** {glm_pred.iloc[0]:.2f} minutes")
        except Exception as e:
            st.error(f"An error occurred during GLM prediction: {e}")
        
        try:
            rf_pred = rf.predict(rf_input_df)
            st.success(f"**Random Forest Predicted Ground Time:** {rf_pred[0]:.2f} minutes")
        except Exception as e:
            st.error(f"An error occurred during Random Forest prediction: {e}")
