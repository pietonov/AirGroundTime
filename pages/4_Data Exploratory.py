######################### Import ##############################
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


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


######################### Data Exploratory ####################

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

# Adjust layout size
fig_full_heatmap.update_layout(
    width=800,
    height=800,
    margin=dict(l=50, r=50, t=50, b=50)
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

