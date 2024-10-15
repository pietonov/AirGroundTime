pip install streamlit

import streamlit as st
import pandas as pd
import plotly.express as px

# Load the summarized data
@st.cache
def load_data():
    return pd.read_csv('OUTPUT/summarized_flight_data.csv')

df = load_data()

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

# Line chart: Total Flights Over Time
st.subheader("Total Flights Over Time")
line_chart_data = filtered_df.groupby('YEAR').agg(total_flights=('total_flights', 'sum')).reset_index()
fig_line = px.line(line_chart_data, x='YEAR', y='total_flights', title='Total Flights Over Time')
st.plotly_chart(fig_line)

# Bar chart: Average Ground Time by Aircraft Configuration
st.subheader("Average Ground Time by Aircraft Configuration")
bar_chart_data = filtered_df.groupby('AIRCRAFT_CONFIG_DESC').agg(average_ground_time=('average_ground_time', 'mean')).reset_index()
fig_bar = px.bar(bar_chart_data, x='AIRCRAFT_CONFIG_DESC', y='average_ground_time', title='Average Ground Time by Aircraft Configuration')
st.plotly_chart(fig_bar)

streamlit run app.py