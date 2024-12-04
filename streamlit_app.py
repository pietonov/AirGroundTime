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

######################### Navigation ###############################

# Add a navigation sidebar with radio buttons
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Docs", "Data Exploratory", "Prediction Apps"])

######################### Sections ###############################

if section == "Docs":
    ######################### Docs ###############################
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

elif section == "Data Exploratory":
    ######################### Data Exploratory ###############################
    
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
    
    # Heatmap
    colormap = st.selectbox("Select a colormap:", options=[
        "RdBu_r", "gray", "Viridis", "Cividis", "Plasma", "Inferno", "Magma"
    ])
    all_options = ["Full Correlation Matrix"] + list(full_correlation_matrix_df.columns)
    selected_feature = st.selectbox('Select a feature to view correlations:', all_options)
    
    if selected_feature == "Full Correlation Matrix":
        st.subheader("Full Correlation Heatmap")
        fig_full_heatmap = px.imshow(
            full_correlation_matrix_df,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=full_correlation_matrix_df.columns,
            y=full_correlation_matrix_df.columns,
            title="Full Feature Correlation Matrix",
            color_continuous_scale=colormap,
            zmin=-1,
            zmax=1,
            width=800,
            height=800
        )
        st.plotly_chart(fig_full_heatmap)
    else:
        selected_correlation = full_correlation_matrix_df[[selected_feature]].transpose()
        st.subheader(f"Correlation Heatmap for {selected_feature}")
        fig_feature_heatmap = px.imshow(
            selected_correlation,
            labels=dict(x="Features", y=selected_feature, color="Correlation"),
            x=full_correlation_matrix_df.columns,
            y=[selected_feature],
            title=f"Correlation Heatmap for {selected_feature}",
            color_continuous_scale=colormap,
            zmin=-1,
            zmax=1,
            width=800,
            height=400
        )
        st.plotly_chart(fig_feature_heatmap)
    
    # Histogram
    st.subheader("Log-Transformed Histogram of GROUND_TIME")
    st.markdown("""
    Ground time is transformed into log for better visualization and analysis due to its long-tail distribution.
                
    The Ground Time is calculated by assuming that RAMP-TO-RAMP = AIR_TIME + GROUND_TIME.
    """)
    num_bins = st.slider("Select number of bins:", min_value=5, max_value=30, value=10)
    bin_step = max(1, len(histogram_summary) // num_bins)
    aggregated_bins = histogram_summary.groupby(histogram_summary.index // bin_step).agg({
        'bin_edges': 'min',
        'frequency': 'sum'
    })
    bin_edges_extended = pd.concat([
        aggregated_bins['bin_edges'], 
        pd.Series([aggregated_bins['bin_edges'].iloc[-1] + 0.1])
    ])
    plt.figure(figsize=(10, 6))
    plt.bar(
        aggregated_bins['bin_edges'], 
        aggregated_bins['frequency'], 
        width=np.diff(bin_edges_extended), 
        color='blue', 
        alpha=0.7, 
        edgecolor='black'
    )
    plt.title('Log-Transformed Distribution of GROUND_TIME')
    plt.xlabel('Log(GROUND_TIME)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    # Boxplot
    st.subheader("Boxplot of LOG_GROUND_TIME")
    st.markdown("""
    The ground time tends to be stable over time; however, there is an indication of cyclical effects, especially in the first quartile, which impacts the IQR and whiskers, opening potential for time-series analysis.
    """)
    df_boxplot = pd.read_csv('DATA/boxplot_summary.csv')
    years_selected = st.multiselect(
        'Select Years', 
        df_boxplot['YEAR'].unique(), 
        default=df_boxplot['YEAR'].unique()
    )
    aircraft_config_selected = st.multiselect(
        'Select Aircraft Configurations', 
        df_boxplot['AIRCRAFT_CONFIG_DESC'].unique(), 
        default=['Freight Configuration', 'Passenger Configuration']
    )
    filtered_df = df_boxplot[
        (df_boxplot['YEAR'].isin(years_selected)) & 
        (df_boxplot['AIRCRAFT_CONFIG_DESC'].isin(aircraft_config_selected))
    ]
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='YEAR', 
        y='LOG_GROUND_TIME', 
        hue='AIRCRAFT_CONFIG_DESC', 
        data=filtered_df, 
        showfliers=False
    )
    plt.ylim(0, 15)
    plt.title('Boxplot of Log Ground Time by Year')
    plt.legend(loc='upper left', title='Aircraft Config')
    plt.xlabel('Year')
    plt.ylabel('Ground Time')
    st.pyplot(plt)
    
    # QQ Plot
    st.subheader("QQ Plot of LOG_GROUND_TIME")
    st.markdown("""
    The QQ Plot does not match any standard distribution. Referring to the histogram, there are two local maxima, suggesting that segmentation of the data might be required. There may be hidden classifications that can be explored.
    """)
    df_qqplot = pd.read_csv('DATA/qq_sample.csv')
    distribution = st.selectbox('Select Distribution for QQ Plot', [
        'norm', 'expon', 'logistic', 'uniform', 'laplace', 'gumbel_r'
    ])
    log_ground_time_sampled = df_qqplot['LOG_GROUND_TIME']
    plt.figure(figsize=(8, 6))
    stats.probplot(log_ground_time_sampled, dist=distribution, plot=plt)
    plt.title(f"QQ Plot Against {distribution.capitalize()} Distribution")
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    st.pyplot(plt)
    
    # Drop Test Results
    st.title("Drop Test Results")
    st.write("""
    Based on the drop test results, we analyze the impact of dropping each variable on RMSEs (Weighted and Unweighted).
    """)
    data = {
        "Dropped Variable": [
            "Full Model (GLM Full)",
            "Drop `UNIQUE_CARRIER`",
            "Drop `DISTANCE`",
            "Drop `LARGE_AIRPORT`",
            "Drop `PASSENGERS`",
            "Drop `IS_WINTER`"
        ],
        "Weighted RMSE": [20.0089, 20.1158, 20.0085, 20.0133, 20.0246, 20.0164],
        "Unweighted RMSE": [94.1800, 94.3295, 94.1891, 94.1909, 94.2371, 94.1860],
        "Difference from Full (Weighted)": [0.0000, 0.1069, -0.0004, 0.0044, 0.0157, 0.0075],
        "Difference from Full (Unweighted)": [0.0000, 0.1495, 0.0091, 0.0109, 0.0571, 0.0060]
    }
    df_drop_test = pd.DataFrame(data)
    st.dataframe(df_drop_test)
    st.subheader("Summary of Findings")
    st.write("""
    - **Retain all variables** since removing any results in noticeably higher RMSEs.
    - **`DISTANCE`** has no significant impact; thus, we will keep it for completeness.
    """)

elif section == "Prediction Apps":
    ######################### Prediction Apps ###############################
    st.title("Predict Ground Time")
    
    # User Input Form on the main page
    st.header("Input Flight Details")
    distance = st.number_input("Distance (miles):", min_value=0, value=500)
    large_airport = st.selectbox("Large Airport:", [0, 1], index=1, format_func=lambda x: 'Yes' if x else 'No')
    has_passengers = st.selectbox("Has Passengers:", [0, 1], index=1, format_func=lambda x: 'Yes' if x else 'No')
    passengers = st.number_input("Number of Passengers:", min_value=0, value=150)
    is_winter = st.selectbox("Winter Season:", [0, 1], index=0, format_func=lambda x: 'Yes' if x else 'No')
    unique_carrier = st.selectbox(
        "Unique Carrier:", 
        options=[
            'American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.',
            'Southwest Airlines Co.', 'Alaska Airlines Inc.', 'Other'
        ]
    )
    
    # Map any unknown carrier to 'Other'
    if unique_carrier not in [
        'American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.',
        'Southwest Airlines Co.', 'Alaska Airlines Inc.', 'Other'
    ]:
        unique_carrier = 'Other'
    
    # Create a DataFrame for input
    input_data = {
        'DISTANCE': [distance],
        'LARGE_AIRPORT': [large_airport],
        'HAS_PASSENGERS': [has_passengers],
        'PASSENGERS': [passengers],
        'IS_WINTER': [is_winter],
        'UNIQUE_CARRIER': [unique_carrier]
    }
    input_df = pd.DataFrame(input_data)
    
    # For GLM Prediction
    glm_input_df = input_df.copy()
    
    # For Random Forest Prediction
    rf_input_df = pd.get_dummies(input_df, columns=['UNIQUE_CARRIER'])
    expected_feature_names = rf.feature_names_in_
    rf_input_df = rf_input_df.reindex(columns=expected_feature_names, fill_value=0)
    
    # Prediction Button
    if st.button("Predict Ground Time"):
        # GLM Prediction
        try:
            glm_pred = glm_full.predict(glm_input_df)
            st.success(f"**GLM Predicted Ground Time:** {glm_pred.iloc[0]:.2f} minutes")
        except Exception as e:
            st.error(f"An error occurred during GLM prediction: {e}")
        
        # Random Forest Prediction
        try:
            rf_pred = rf.predict(rf_input_df)
            st.success(f"**Random Forest Predicted Ground Time:** {rf_pred[0]:.2f} minutes")
        except Exception as e:
            st.error(f"An error occurred during Random Forest prediction: {e}")
