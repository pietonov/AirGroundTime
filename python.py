# %%
# Import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# %%
df_carrier = []

files = [
    '2014_T_T100D_SEGMENT_ALL_CARRIER.csv', '2015_T_T100D_SEGMENT_ALL_CARRIER.csv', 
    '2016_T_T100D_SEGMENT_ALL_CARRIER.csv', '2017_T_T100D_SEGMENT_ALL_CARRIER.csv', 
    '2018_T_T100D_SEGMENT_ALL_CARRIER.csv', '2019_T_T100D_SEGMENT_ALL_CARRIER.csv', 
    '2020_T_T100D_SEGMENT_ALL_CARRIER.csv', '2021_T_T100D_SEGMENT_ALL_CARRIER.csv', 
    '2022_T_T100D_SEGMENT_ALL_CARRIER.csv', '2023_T_T100D_SEGMENT_ALL_CARRIER.csv', 
    '2024_T_T100D_SEGMENT_ALL_CARRIER.csv'
]

folder_path = 'DATA'

df_carrier = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in files], ignore_index=True)
# df_carrier.replace(0, np.nan, inplace=True)

df_carrier.head()

# %%
# Validation

# Count the number of rows where DEPARTURES_PERFORMED is 0 and AIR_TIME > 0
INVALID_DEPARTURE_PERFORMED = ((df_carrier['DEPARTURES_PERFORMED'] == 0) & (df_carrier['AIR_TIME'] > 0)).sum()
print(f"Number of invalid DEPARTURES_PERFORMED: {INVALID_DEPARTURE_PERFORMED}")

# If DEPARTURES_PERFORMED is 0 and AIR_TIME > 0, set DEPARTURES_PERFORMED to 1
df_carrier.loc[(df_carrier['DEPARTURES_PERFORMED'] == 0) & (df_carrier['AIR_TIME'] > 0), 'DEPARTURES_PERFORMED'] = 1

# Else if DEPARTURES_PERFORMED is 0 then remove the rows as there is no AIR_TIME.
df_carrier = df_carrier[~((df_carrier['DEPARTURES_PERFORMED'] == 0) & (df_carrier['AIR_TIME'] == 0))]

# If DISTANCE > 0 and AIR_TIME = 0 then AIR_TIME is missing
df_carrier.loc[(df_carrier['DISTANCE'] > 0) & (df_carrier['AIR_TIME'] == 0), 'AIR_TIME'] = np.nan

# If DISTANCE > 0 and RAMP_TO_RAMP = 0 then RAMP_TO_RAMP is missing
df_carrier.loc[(df_carrier['DISTANCE'] > 0) & (df_carrier['RAMP_TO_RAMP'] == 0), 'RAMP_TO_RAMP'] = np.nan

# Calculate GROUND_TIME as the difference between RAMP_TO_RAMP and AIR_TIME
df_carrier['GROUND_TIME'] = df_carrier['RAMP_TO_RAMP'] - df_carrier['AIR_TIME']

# Count the number of rows where RAMP_TO_RAMP is less than AIR_TIME
INVALID_AIR_TIME = (df_carrier['RAMP_TO_RAMP'] < df_carrier['AIR_TIME']).sum()
print(f"Number of invalid AIR_TIME: {INVALID_AIR_TIME}")



# %%
# Check if GROUND_TIME has missing values (NaN)
missing_data = df_carrier['GROUND_TIME'].isnull()

# Display rows where GROUND_TIME is missing
df_missing_ground_time = df_carrier[missing_data]

# Show the missing data
df_missing_ground_time.head()

# %%
# Import lookup tables
folder_path = 'TABLES'
tab_aircraft_config = pd.read_csv(os.path.join(folder_path, 'L_AIRCRAFT_CONFIG.csv'))
tab_aircraft_type = pd.read_csv(os.path.join(folder_path, 'L_AIRCRAFT_TYPE.csv'))
tab_airport_id = pd.read_csv(os.path.join(folder_path, 'L_AIRPORT_ID.csv'))
tab_airport = pd.read_csv(os.path.join(folder_path, 'L_AIRPORT.csv'))
tab_distance_group = pd.read_csv(os.path.join(folder_path, 'L_DISTANCE_GROUP_500.csv'))
tab_service_class = pd.read_csv(os.path.join(folder_path, 'L_SERVICE_CLASS.csv'))
tab_unique_carriers = pd.read_csv(os.path.join(folder_path, 'L_UNIQUE_CARRIERS.csv'))


# %%
# Import df_aiport (predictor dataset)

df_airport = []
folder_path = 'DATA'
df_airport = pd.read_csv(os.path.join(folder_path, 'airports.csv'))

df_airport.head()


# %%
# Filter data with IATA only and US only
df_filtered = df_airport[(df_airport['iata_code'].notna()) & (df_airport['iso_country'] == 'US')]

# Create a lookup table for predictor
airport_tab_fac = df_filtered[['iata_code', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'type']]

# %%
# Merge df_carrier with all columns from airport_tab_fac
df_carrier = df_carrier.merge(airport_tab_fac, 
                              left_on='ORIGIN', 
                              right_on='iata_code', 
                              how='left')

# Drop the 'iata_code' column
df_carrier.drop(columns=['iata_code'], inplace=True)

# %%
# Filter Michigan only
df_michigan = df_carrier[df_carrier['ORIGIN_STATE_NM'] == 'Michigan']

# Create a pivot table for Michigan
pivot_table_michigan = pd.pivot_table(df_michigan, 
                                      values='DEPARTURES_PERFORMED', 
                                      index='YEAR', 
                                      columns='MONTH', 
                                      aggfunc='sum')

# Display the pivot table
pivot_table_michigan = pivot_table_michigan / 1000

# %%
# Create the heatmap
plt.figure(figsize=(10, 6))  # Set the figure size
sns.heatmap(pivot_table_michigan, annot=True, fmt=".0f", cmap='cividis', linewidths=.5)

# Add titles and labels
plt.title('Heatmap of Departures (in \'000) by Year and Month (Michigan)')
plt.xlabel('Month')
plt.ylabel('Year')

# Show the heatmap
plt.show()

# %%
# Step 1: Summary statistics for GROUND_TIME
ground_time_summary = df_carrier['GROUND_TIME'].describe()
print("Summary Statistics for GROUND_TIME:")
print(ground_time_summary)

# Step 2: Check for missing values
missing_ground_time = df_carrier['GROUND_TIME'].isnull().sum()
print(f"\nMissing values in GROUND_TIME: {missing_ground_time}")

# Step 3: Plotting the distribution of GROUND_TIME
plt.figure(figsize=(10, 6))

# Histogram and KDE plot for GROUND_TIME
sns.histplot(df_carrier['GROUND_TIME'], kde=True, bins=30, color='blue')
plt.title('Distribution of GROUND_TIME')
plt.xlabel('GROUND_TIME')
plt.ylabel('Frequency')

plt.show()

# Step 4: Boxplot to detect outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_carrier['GROUND_TIME'], color='green')
plt.title('Boxplot of GROUND_TIME')
plt.xlabel('GROUND_TIME')

plt.show()

# %%
# Log transform the GROUND_TIME to reduce skewness
df_carrier['LOG_GROUND_TIME'] = np.log1p(df_carrier['GROUND_TIME'])  # log1p to handle 0 values



# %%
# Plot the log-transformed GROUND_TIME distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_carrier['LOG_GROUND_TIME'], kde=True, bins=30, color='blue')
plt.title('Log-Transformed Distribution of GROUND_TIME')
plt.xlabel('Log(GROUND_TIME)')
plt.ylabel('Frequency')

plt.show()

# %%
# Translate aircraft_config using tab_aircraft_config
tab_aircraft_config.rename(columns={'Description': 'AIRCRAFT_CONFIG_DESC'}, inplace=True)

df_carrier = df_carrier.merge(tab_aircraft_config, 
                              left_on='AIRCRAFT_CONFIG', 
                              right_on='Code', 
                              how='left')

df_carrier.drop(columns=['Code'], inplace=True)

# %%
# Create the boxplot for LOG_GROUND_TIME by YEAR, divided by AIRCRAFT_CONFIG
plt.figure(figsize=(12, 6))

sns.boxplot(data=df_carrier, x='YEAR', y='LOG_GROUND_TIME', hue='AIRCRAFT_CONFIG_DESC')
plt.ylim(0, 15)
plt.title('Boxplot of Log Ground Time by Year')
plt.legend(loc='upper left', title='Aircraft Config')
plt.xlabel('Year')
plt.ylabel('Ground Time')
plt.show()

# %%
# Select only numerical columns from the DataFrame
df_numerical = df_carrier.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix for the numerical columns
correlation_matrix = df_numerical.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='cividis', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()


# %%
# Filter the correlation matrix to only include the 'LOG_GROUND_TIME' row
predictor_candidate = ['PAYLOAD', 'SEATS', 'PASSENGERS', 'FREIGHT', 'MAIL', 'DISTANCE', 'latitude_deg', 'elevation_ft', 'longitude_deg']
log_ground_time_corr = correlation_matrix.loc[['LOG_GROUND_TIME'], predictor_candidate]


# Create a heatmap for the filtered data
plt.figure(figsize=(12, 1)) 
#sns.heatmap(log_ground_time_corr.T, annot=False, cmap='cividis', linewidths=0.5, cbar=True)
sns.heatmap(log_ground_time_corr, annot=True, cmap='coolwarm', linewidths=0.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlation Heatmap for LOG_GROUND_TIME')
plt.show()


