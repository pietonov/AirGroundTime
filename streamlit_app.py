import streamlit as st
import pandas as pd
import plotly.express as px

# Load the summarized data
def load_data():
    return pd.read_csv('DATA/summarized_flight_data.csv')

# Load the full correlation matrix data
def load_correlation_matrix():
    return pd.read_csv('DATA/correlation_matrix.csv')

df = load_data()
full_correlation_matrix_df = load_correlation_matrix()

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

# Interactive Heatmap for the Full Correlation Matrix
st.subheader("Full Correlation Heatmap")
fig_heatmap = px.imshow(
    full_correlation_matrix_df,
    labels=dict(x="Features", y="Features", color="Correlation"),
    x=full_correlation_matrix_df.columns,
    y=full_correlation_matrix_df.columns,
    title="Full Feature Correlation Matrix",
    width=800,
    height=800 
)
fig_heatmap.update_layout(margin=dict(l=100, r=100, b=100, t=100))
st.plotly_chart(fig_heatmap)






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

### Data Description
- **YEAR:** Year of data
- **AIRCRAFT_CONFIG_DESC:** Type of aircraft: Passenger Flight, Freight, Sea, etc.
- **total_flights:** Total number of flights.
- **average_ground_time:** The average ground time.
""")
