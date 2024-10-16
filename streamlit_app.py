import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the summarized data
def load_data():
    return pd.read_csv('DATA/summarized_flight_data.csv')

# Load the full correlation matrix data
def load_correlation_matrix():
    return pd.read_csv('DATA/correlation_matrix.csv')

# Load the histogram summary data for GROUND_TIME
def load_histogram_summary():
    return pd.read_csv('DATA/histogram_summary_ground_time.csv')

df = load_data()
full_correlation_matrix_df = load_correlation_matrix()
histogram_summary = load_histogram_summary()

# Streamlit App Title
st.title("Flight Statistics Interactive Visualization")

# Sidebar Filters
year_filter = st.sidebar.multiselect(
    "Select Year(s)", options=df['YEAR'].unique(), default=df['YEAR'].unique()
)
aircraft_filter = st.sidebar.multiselect(
    "Select Aircraft Configuration(s)", options=df['AIRCRAFT_CONFIG_DESC'].unique(), default=df['AIRCRAFT_CONFIG_DESC'].unique()
)

# Filtered Data
filtered_df = df[(df['YEAR'].isin(year_filter)) & (df['AIRCRAFT_CONFIG_DESC'].isin(aircraft_filter))]

# Display the filtered dataset
st.dataframe(filtered_df)

# Line chart: Total Flights Over Time, differentiated by Aircraft Configuration
st.subheader("Total Flights Over Time by Aircraft Configuration")
fig_line = px.line(
    filtered_df, 
    x='YEAR', 
    y='total_flights', 
    color='AIRCRAFT_CONFIG_DESC',  # Differentiate lines by aircraft config
    title='Total Flights Over Time by Aircraft Configuration'
)
st.plotly_chart(fig_line)

# Bar chart: Average Ground Time by Aircraft Configuration
st.subheader("Average Ground Time by Aircraft Configuration")
bar_chart_data = filtered_df.groupby('AIRCRAFT_CONFIG_DESC').agg(average_ground_time=('average_ground_time', 'mean')).reset_index()
fig_bar = px.bar(bar_chart_data, x='AIRCRAFT_CONFIG_DESC', y='average_ground_time', title='Average Ground Time by Aircraft Configuration')
st.plotly_chart(fig_bar)

# Dropdown for colormap selection
colormap = st.selectbox("Select a colormap:", options=["RdBu_r", "gray", "Viridis", "Cividis", "Plasma", "Inferno", "Magma"])

# Add "Full Correlation Matrix" as an option in the dropdown
all_options = ["Full Correlation Matrix"] + list(full_correlation_matrix_df.columns)
selected_feature = st.selectbox('Select a feature to view correlations:', all_options)

# Display either the full correlation matrix or the specific feature's correlations
if selected_feature == "Full Correlation Matrix":
    # Full Correlation Matrix
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
    # Show correlations for the selected feature
    selected_correlation = full_correlation_matrix_df[[selected_feature]].transpose()

    # Interactive Heatmap for the selected feature
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

# INTERACTIVE HISTOGRAM SECTION
st.subheader("Log-Transformed Histogram of GROUND_TIME")

# Add a slider for binning (interactive control)
num_bins = st.slider("Select number of bins:", min_value=5, max_value=30, value=10)

# Re-aggregate bins dynamically based on user-selected bin count
bin_step = max(1, len(histogram_summary) // num_bins)  # Determine how many bins to aggregate
aggregated_bins = histogram_summary.groupby(histogram_summary.index // bin_step).agg({
    'bin_edges': 'min',  # Use the left edge of the first bin in each aggregated group
    'frequency': 'sum'   # Sum the frequencies of the bins being aggregated
})

# Correct the error using pd.concat
bin_edges_extended = pd.concat([aggregated_bins['bin_edges'], pd.Series([aggregated_bins['bin_edges'].iloc[-1] + 0.1])])

# Plot the dynamically aggregated histogram
plt.figure(figsize=(10, 6))
plt.bar(aggregated_bins['bin_edges'], aggregated_bins['frequency'], width=np.diff(bin_edges_extended), color='blue', alpha=0.7, edgecolor='black')
plt.title('Log-Transformed Distribution of GROUND_TIME')
plt.xlabel('Log(GROUND_TIME)')
plt.ylabel('Frequency')
st.pyplot()









# Docs

st.title("Flight Statistics Interactive Documentation")

st.markdown("""
## Documentation

### Purpose
This app visualizes flight statistics in United States. User can explore total number of flights and the average of ground time based on aircraft configuration type.

### Instructions
- **Select Year:** Use the sidebar to filter the data by year.
- **Select Aircraft Configuration:** select the aircraft config in the sidebar.
- **Interactive Charts:** User can hover on data point to see the detail, also can zoom in and out depend on usage.
- **Heatmap:** Use dropdown to choose the colormap and the features that you want to focus on.

### Data Description
- **YEAR:** Year of data
- **AIRCRAFT_CONFIG_DESC:** Type of aircraft: Passenger Flight, Freight, Sea, etc.
- **total_flights:** Total number of flights.
- **average_ground_time:** The average ground time.
""")
