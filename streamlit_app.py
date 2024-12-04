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
section = st.sidebar.selectbox(
    "Go to Section:",
    ["Docs", "Data Exploratory", "Prediction Apps"]
)

######################### Documentation #######################
if section == "Docs":
    st.title("Flight Statistics Interactive Documentation")
    st.markdown("""
    ## Purpose
    This app visualizes flight statistics in the United States. Users can explore total flights and average ground time by aircraft configuration.

    ## Instructions
    - Use the navigation menu to switch sections.
    - Explore various visualizations and predictive capabilities.

    ## Data Description
    - **YEAR**: Year of data.
    - **AIRCRAFT_CONFIG_DESC**: Type of aircraft: Passenger, Freight, etc.
    - **total_flights**: Total number of flights.
    - **average_ground_time**: The average ground time.
    """)

######################### Data Exploratory ####################
elif section == "Data Exploratory":
    st.title("Flight Data Exploration")
    df = load_data()
    full_correlation_matrix_df = load_correlation_matrix()
    histogram_summary = load_histogram_summary()
    boxplot_data = load_boxplot_data()
    qqplot_data = load_qqplot_data()

    # Filters
    st.subheader("Filters")
    year_filter = st.multiselect("Select Year(s):", options=df['YEAR'].unique(), default=df['YEAR'].unique())
    aircraft_filter = st.multiselect("Select Aircraft Configuration(s):", options=df['AIRCRAFT_CONFIG_DESC'].unique(), default=df['AIRCRAFT_CONFIG_DESC'].unique())
    filtered_df = df[(df['YEAR'].isin(year_filter)) & (df['AIRCRAFT_CONFIG_DESC'].isin(aircraft_filter))]

    # Filtered Data
    st.write("### Filtered Dataset")
    st.dataframe(filtered_df)

    # Line Chart
    st.write("### Total Flights Over Time")
    fig_line = px.line(filtered_df, x='YEAR', y='total_flights', color='AIRCRAFT_CONFIG_DESC', title="Total Flights Over Time by Aircraft Configuration")
    st.plotly_chart(fig_line)

    # Bar Chart
    st.write("### Average Ground Time by Aircraft Configuration")
    avg_ground_time = filtered_df.groupby('AIRCRAFT_CONFIG_DESC').agg(average_ground_time=('average_ground_time', 'mean')).reset_index()
    fig_bar = px.bar(avg_ground_time, x='AIRCRAFT_CONFIG_DESC', y='average_ground_time', title="Average Ground Time by Aircraft Configuration")
    st.plotly_chart(fig_bar)

    # Heatmap
    st.write("### Correlation Heatmap")
    colormap = st.selectbox("Select a colormap:", options=["RdBu_r", "gray", "Viridis", "Cividis", "Plasma", "Inferno", "Magma"])
    fig_full_heatmap = px.imshow(
        full_correlation_matrix_df,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=full_correlation_matrix_df.columns,
        y=full_correlation_matrix_df.columns,
        title="Full Feature Correlation Matrix",
        color_continuous_scale=colormap,
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_full_heatmap)

    # Histogram
    st.write("### Log-Transformed Histogram of GROUND_TIME")
    num_bins = st.slider("Select number of bins:", min_value=5, max_value=30, value=10)
    bin_step = max(1, len(histogram_summary) // num_bins)
    aggregated_bins = histogram_summary.groupby(histogram_summary.index // bin_step).agg({'bin_edges': 'min', 'frequency': 'sum'})
    plt.figure(figsize=(10, 6))
    plt.bar(aggregated_bins['bin_edges'], aggregated_bins['frequency'], width=0.1, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Log-Transformed Distribution of GROUND_TIME')
    plt.xlabel('Log(GROUND_TIME)')
    plt.ylabel('Frequency')
    st.pyplot()

    # Boxplot
    st.write("### Boxplot of LOG_GROUND_TIME")
    years_selected = st.multiselect('Select Years', boxplot_data['YEAR'].unique(), default=boxplot_data['YEAR'].unique())
    aircraft_config_selected = st.multiselect('Select Aircraft Configurations', boxplot_data['AIRCRAFT_CONFIG_DESC'].unique(), default=boxplot_data['AIRCRAFT_CONFIG_DESC'].unique())
    filtered_boxplot = boxplot_data[(boxplot_data['YEAR'].isin(years_selected)) & (boxplot_data['AIRCRAFT_CONFIG_DESC'].isin(aircraft_config_selected))]
    sns.boxplot(x='YEAR', y='LOG_GROUND_TIME', hue='AIRCRAFT_CONFIG_DESC', data=filtered_boxplot, showfliers=False)
    plt.title('Boxplot of Log Ground Time by Year')
    st.pyplot()

    # QQ Plot
    st.write("### QQ Plot of LOG_GROUND_TIME")
    distribution = st.selectbox("Select Distribution for QQ Plot:", ['norm', 'expon', 'logistic', 'uniform', 'laplace', 'gumbel_r'])
    plt.figure(figsize=(8, 6))
    stats.probplot(qqplot_data['LOG_GROUND_TIME'], dist=distribution, plot=plt)
    plt.title(f"QQ Plot Against {distribution.capitalize()} Distribution")
    st.pyplot()

######################### Prediction Apps #####################
elif section == "Prediction Apps":
    st.title("Predict Ground Time")

    # User Input Form
    with st.form("prediction_form"):
        st.header("Input Flight Details")
        distance = st.number_input("Distance (miles):", min_value=0, value=500)
        large_airport = st.selectbox("Large Airport:", [1, 0], format_func=lambda x: "Yes" if x else "No")
        has_passengers = st.selectbox("Has Passengers:", [1, 0], format_func=lambda x: "Yes" if x else "No")
        passengers = st.number_input("Number of Passengers:", min_value=0, value=150, disabled=not has_passengers)
        is_winter = st.selectbox("Winter Season:", [1, 0], format_func=lambda x: "Yes" if x else "No")
        unique_carrier = st.selectbox("Unique Carrier:", ['American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.', 'Southwest Airlines Co.', 'Alaska Airlines Inc.', 'Other'])

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
