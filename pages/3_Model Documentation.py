######################### Import ##############################
import streamlit as st
import pandas as pd
import os


# Sidebar Table of Contents
st.sidebar.title("Table of Contents")
st.sidebar.markdown("""
- [Data Standardization](#data-standardization)
- [Predictor Variable (Initial Assessment)](#predictor-variable-initial-assessment)
- [Feature Engineering](#feature-engineering)
- [Optimization](#optimization)
- [Encoding](#encoding)
- [Model Train](#model-train)
- [Modeling](#modeling)
  - [Baseline](#baseline)
  - [GLM Model Summary](#glm-model-summary)
  - [GLM Drop Test Results](#glm-drop-test-results)
  - [Residual Plot](#residual-plot)
  - [GLM Full Model Summary](#glm-full-model-summary)
- [Random Forest](#random-forest)
  - [Random Forest Testing Results](#random-forest-testing-results)
  - [Random Forest Drop Testing Results](#random-forest-drop-testing-results)
  - [Feature Importance](#feature-importance)
  - [Partial Dependence Plots](#partial-dependence-plots)
- [Unique Carrier Analysis](#unique-carrier-analysis)
""", unsafe_allow_html=True)


st.title("Model Documentation")

# Data Standardization
st.header("Data Standardization")
st.markdown("""
<a id="data-standardization"></a>
Note that some variables, such as **PASSENGERS**, **FREIGHT**, **MAIL**, **RAMP_TO_RAMP**, and **AIR_TIME** should be divided by **DEPARTURE_PERFORMED**. Dividing by the number of departures gives the average time per flight, to ensure consistent comparisons between periods or across routes.

**Focus on DTW only** since it encountered memory overload on several occasions during GLM modeling.
""", unsafe_allow_html=True)

# Predictor Variable (Initial Assessment)
st.header("Predictor Variable (Initial Assessment)")
st.markdown("""
<a id="predictor-variable-initial-assessment"></a>
Based on the data, we can expect which variables might have an impact. Below is the initial list of candidate variables:

- **UNIQUE_CARRIER**  
    Airlines might have their own strategies, even within the same company. For example, Singapore Airlines (Premium) owns Scoot Airlines (Low-cost). Low-cost carriers may have shorter ground-time compared to premium ones. Freight companies like UPS or FedEx might have longer ground time.

- **DISTANCE**  
    Longer flights require larger airplanes which can carry more fuel and require more loads to be efficient, thus affecting ground time.

- **LARGE_AIRPORT**  
    Larger airports might have better equipment, but conversely, they also might have high traffic.

- **PASSENGERS**  
    More passengers mean more complications.

- **FREIGHTS**  
    Similar impact as passengers.

- **MAIL**  
    Similar impact as freights.
""", unsafe_allow_html=True)

# Feature Engineering
st.header("Feature Engineering")
st.markdown("""
<a id="feature-engineering"></a>
New boolean variables were added to improve the GLM model:

- **HAS_PASSENGERS**
- **HAS_FREIGHTS**
- **HAS_MAIL**
""", unsafe_allow_html=True)

# Optimization
st.header("Optimization")
st.markdown("""
<a id="optimization"></a>
Since the GLM model didn't run efficiently, we optimized the model by downcasting numerical columns.
Also, reduced the **UNIQUE_CARRIER** number to only the top 20.
""", unsafe_allow_html=True)

# Encoding
st.header("Encoding")
st.markdown("""
<a id="encoding"></a>
**UNIQUE_CARRIER** and **LARGE_AIRPORT** need encoding. Thus, one-hot encoding was applied for **UNIQUE_CARRIER** and binary encoding for **LARGE_AIRPORT**. One-hot encoding has been chosen over label and target encoding based on the following cons:

- **One-hot encoding**  
    Cons: Takes up more space.

- **Label encoding**  
    Cons: Assumes order which may not exist.

- **Target encoding**  
    Cons: Risk of data leakage.
""", unsafe_allow_html=True)

# Model Train
st.header("Model Train")
st.markdown("""
<a id="model-train"></a>
The 2024 data was omitted since the cohort is incomplete.  
**Train data**: 2014 - 2022  
**Test data**: 2023  

- Training data: 2014-2022 (n=126,351)
- Test data: 2023 (n=11,171)
""", unsafe_allow_html=True)

# Modeling
st.header("Modeling")

## Baseline
st.subheader("Baseline")
st.markdown("Intercept-only model as a benchmark.")
st.markdown("""<a id="baseline"></a>
**Weighted Intercept-Only Model Summary:**

```
Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:            GROUND_TIME   No. Observations:               126351
Model:                            GLM   Df Residuals:                  2866222
Model Family:                Gaussian   Df Model:                            0
Link Function:               Identity   Scale:                          244.12
Method:                          IRLS   Log-Likelihood:            -1.1946e+07
Date:                Tue, 03 Dec 2024   Deviance:                   6.9970e+08
Time:                        10:45:16   Pearson chi2:                 7.00e+08
No. Iterations:                     3   Pseudo R-squ. (CS):          1.415e-12
Covariance Type:            nonrobust                                         
==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     24.8669      0.009   2694.487      0.000      24.849      24.885
==============================================================================
```

**Unweighted Intercept-Only Model Summary:**

```
Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:            GROUND_TIME   No. Observations:               126351
Model:                            GLM   Df Residuals:                   126350
Model Family:                Gaussian   Df Model:                            0
Link Function:               Identity   Scale:                          5008.6
Method:                          IRLS   Log-Likelihood:            -7.1747e+05
Date:                Tue, 03 Dec 2024   Deviance:                   6.3284e+08
Time:                        10:45:16   Pearson chi2:                 6.33e+08
No. Iterations:                     3   Pseudo R-squ. (CS):          3.132e-11
Covariance Type:            nonrobust                                         
==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     23.0890      0.199    115.967      0.000      22.699      23.479
==============================================================================
```
""", unsafe_allow_html=True)

# GLM Model Summary
st.subheader("GLM Model Summary")
st.markdown("""<a id="glm-model-summary"></a>
| GLM Model         | Variables Added          | Deviation (Start -> End) | Improvement | Status                               |
|-------------------|--------------------------|--------------------------|-------------|--------------------------------------|
| GLM 0             | INTERCEPT_ONLY           | 20.1269                  | N/A         | Initial model                        |
| GLM 1             | UNIQUE_CARRIER           | 20.1269 -> 20.0395       | -0.0874     | Accepted                             |
| GLM 2             | DISTANCE                 | 20.0395 -> 20.0352       | -0.0043     | Accepted                             |
| GLM 3             | LARGE_AIRPORT            | 20.0352 -> 20.0296       | -0.0056     | Accepted                             |
| GLM 4A            | HAS_PASSENGERS           | 20.0296 -> 20.0290       | -0.0006     | Rejected (insignificant)             |
| GLM 4B            | PASSENGERS               | 20.0296 -> 20.0164       | -0.0132     | Accepted                             |
| GLM 5A            | HAS_FREIGHTS             | 20.0164 -> 20.0139       | -0.0025     | Rejected (insignificant)             |
| GLM 5B            | FREIGHTS                 | 20.0164 -> 20.0125       | -0.0039     | Rejected (insignificant)             |
| GLM 5C            | HAS_MAIL                 | 20.0164 -> 20.0151       | -0.0013     | Rejected (insignificant)             |
| GLM 5D            | MAIL                     | 20.0164 -> 20.0170       | +0.0006     | Rejected (increasing)                |
| GLM 5E            | MONTH                    | 20.0164 -> 20.0110       | -0.0054     | Rejected (isolated winter accepted)  |
| GLM 5F            | IS_WINTER                | 20.0164 -> 20.0089       | -0.0075     | Accepted (selected isolation)        |
| GLM 5G            | PEAK_SEASON              | 20.0164 -> 20.0162       | -0.0002     | Rejected (isolated winter accepted)  |
| **GLM FULL MODEL**| **FULL + HAS_PASSENGERS**| **20.1269 -> 20.0091**   | **-0.1178** | **Final model**                      |
""", unsafe_allow_html=True)

st.markdown("""
**Note:** `HAS_PASSENGERS` is added in the final GLM model to support the `PASSENGERS` variable even though it slightly worsens the model as of right now.  
Based on the test above, we accept `UNIQUE_CARRIER`, `DISTANCE`, `LARGE_AIRPORT`, `HAS_PASSENGERS`, `PASSENGERS`, and `IS_WINTER`.
""", unsafe_allow_html=True)

# Dropped Variable Table
st.subheader("GLM Drop Test Results")
st.markdown("""<a id="glm-drop-test-results"></a>""", unsafe_allow_html=True)
df_glm_drop = pd.DataFrame({
    'Dropped Variable': ['Full Model (GLM Full)', 'Drop UNIQUE_CARRIER', 'Drop DISTANCE', 'Drop LARGE_AIRPORT', 'Drop HAS_PASSENGERS', 'Drop PASSENGERS', 'Drop IS_WINTER', 'Intercept-Only Model'],
    'Weighted RMSE': [20.0091, 20.1064, 20.0086, 20.0135, 20.0089, 20.0239, 20.0165, 20.1269],
    'Unweighted RMSE': [94.1825, 94.3032, 94.1913, 94.1938, 94.1800, 94.2291, 94.1878, 94.2967],
    'Difference from Full (Weighted)': ['0.0000', '+0.0973', '-0.0005', '+0.0044', '-0.0002', '+0.0148', '+0.0074', '+0.1178'],
    'Difference from Full (Unweighted)': ['0.0000', '+0.1207', '+0.0088', '+0.0113', '-0.0025', '+0.0466', '+0.0053', '+0.1142']
})
st.table(df_glm_drop)

st.markdown("""
Based on the drop test above, we retain all the variables since removing any of them results in noticeably higher RMSEs. For `DISTANCE`, even though it has minimal impact, it's reasonable to keep it as a significant feature.
""", unsafe_allow_html=True)

# Residual Plot
st.subheader("Residual Plot")
st.markdown("""<a id=residual-plot""></a>""", unsafe_allow_html=True)
residual_plot_path = 'DATA/residual.png'
if os.path.exists(residual_plot_path):
    st.image(residual_plot_path, caption='Residual Plot')
else:
    st.warning('Residual plot not found.')

st.markdown("""
Based on the residual plot above, it indicates that the GLM has similarly distributed errors across the range of predictions between 15 to 35 minutes. However, there are significant outliers in the residuals. This suggests that the model has difficulties fitting some of the observations.
""", unsafe_allow_html=True)

# GLM Full Model Summary
st.subheader("GLM Full Model Summary")
st.markdown("""<a id="glm-full-model-summary"></a>
```
==============================================================================
                Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:            GROUND_TIME   No. Observations:               126351
Model:                            GLM   Df Residuals:                  2866197
Model Family:                Gaussian   Df Model:                           25
Link Function:               Identity   Scale:                          239.03
Method:                          IRLS   Log-Likelihood:            -1.1916e+07
Date:                Wed, 04 Dec 2024   Deviance:                   6.8512e+08
Time:                        12:50:01   Pearson chi2:                 6.85e+08
No. Iterations:                     3   Pseudo R-squ. (CS):             0.3829
Covariance Type:            nonrobust                                         
=============================================================================================================================
                                                                coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------------------------
Intercept                                                    22.8905      0.238     96.283      0.000      22.425      23.356
UNIQUE_CARRIER[T.American Airlines Inc.]                      0.9752      0.188      5.177      0.000       0.606       1.344
UNIQUE_CARRIER[T.Chautauqua Airlines Inc.]                    0.9024      0.208      4.340      0.000       0.495       1.310
UNIQUE_CARRIER[T.Compass Airlines]                            0.4917      0.220      2.236      0.025       0.061       0.923
UNIQUE_CARRIER[T.Delta Air Lines Inc.]                       -3.1608      0.181    -17.449      0.000      -3.516      -2.806
UNIQUE_CARRIER[T.Endeavor Air Inc.]                          -0.3236      0.184     -1.756      0.079      -0.685       0.038
UNIQUE_CARRIER[T.Envoy Air]                                   6.6744      0.208     32.080      0.000       6.267       7.082
UNIQUE_CARRIER[T.ExpressJet Airlines LLC d/b/a aha!]         -1.4582      0.187     -7.817      0.000      -1.824      -1.093
UNIQUE_CARRIER[T.Federal Express Corporation]                -2.9229      0.255    -11.462      0.000      -3.423      -2.423
UNIQUE_CARRIER[T.Frontier Airlines Inc.]                     -1.9981      0.216     -9.230      0.000      -2.422      -1.574
UNIQUE_CARRIER[T.GoJet Airlines LLC d/b/a United Express]     0.5893      0.188      3.134      0.002       0.221       0.958
UNIQUE_CARRIER[T.JetBlue Airways]                            -1.9616      0.212     -9.274      0.000      -2.376      -1.547
UNIQUE_CARRIER[T.Other]                                      -2.5068      0.191    -13.110      0.000      -2.882      -2.132
UNIQUE_CARRIER[T.PSA Airlines Inc.]                           4.7844      0.212     22.546      0.000       4.368       5.200
UNIQUE_CARRIER[T.Republic Airline]                            3.4161      0.188     18.136      0.000       3.047       3.785
UNIQUE_CARRIER[T.Shuttle America Corp.]                      -0.7460      0.200     -3.739      0.000      -1.137      -0.355
UNIQUE_CARRIER[T.SkyWest Airlines Inc.]                       0.8888      0.184      4.817      0.000       0.527       1.250
UNIQUE_CARRIER[T.Southwest Airlines Co.]                     -7.3822      0.188    -39.316      0.000      -7.750      -7.014
UNIQUE_CARRIER[T.Spirit Air Lines]                           -3.9735      0.185    -21.529      0.000      -4.335      -3.612
UNIQUE_CARRIER[T.United Air Lines Inc.]                      -0.7153      0.210     -3.414      0.001      -1.126      -0.305
UNIQUE_CARRIER[T.United Parcel Service]                      -5.7845      0.276    -20.935      0.000      -6.326      -5.243
DISTANCE                                                     -0.0004   2.59e-05    -15.410      0.000      -0.000      -0.000
LARGE_AIRPORT                                                 1.1324      0.029     38.877      0.000       1.075       1.190
HAS_PASSENGERS                                               -0.6246      0.152     -4.117      0.000      -0.922      -0.327
PASSENGERS                                                    0.0321      0.000     81.454      0.000       0.031       0.033
IS_WINTER                                                     3.0632      0.025    123.064      0.000       3.014       3.112
=============================================================================================================================
```
""", unsafe_allow_html=True)

st.markdown("""
**1. Model Fitness**

- **Pseudo R-squared**: At 0.3829, it shows a significant improvement over the intercept-only model, but the model still doesn't explain the majority of the variability in the dependent variable.
- **Chi-squared**: Large numbers are expected due to the vast amount of samples.

**2. Predictors**

- **UNIQUE_CARRIER**: Most carrier coefficients are highly significant (P < 0.05), with varying impacts on `GROUND_TIME`.
    - American Airlines Inc. increases `GROUND_TIME` by 0.9752 minutes.
    - Southwest Airlines Co. reduces it significantly by -7.3822 minutes.
    - Envoy Air has a substantial positive effect of +6.6744 minutes.
    - Carriers like Delta Air Lines Inc. (-3.1608) and Spirit Air Lines (-3.9735) significantly reduce `GROUND_TIME`.
    - Other carriers tend to reduce `GROUND_TIME` by -2.5068 minutes.
- **DISTANCE**: Small but significant negative effect (-0.0004). Distances are weakly correlated with `GROUND_TIME`, possibly due to different operational practices or scheduling priorities for long-haul flights.
- **LARGE_AIRPORT**: Larger airports tend to have longer `GROUND_TIME` (+1.1324 minutes), possibly due to increased traffic, congestion, or more complex procedures.
- **IS_WINTER**: Winter months contribute to longer `GROUND_TIME` (+3.0632 minutes), likely due to de-icing procedures, bad weather, or additional safety checks.
- **PASSENGERS**: Each passenger increases `GROUND_TIME` by 0.0321 minutes, reflecting boarding and deplaning times.
- **HAS_PASSENGERS**: Flights without passengers (Freight) have shorter `GROUND_TIME` by -0.6246 minutes.
""", unsafe_allow_html=True)

# Random Forest
st.header("Random Forest")
st.markdown("""<a id="random-forest"></a>""", unsafe_allow_html=True)

## Random Forest Testing Results
st.subheader("Random Forest Testing Results")
st.markdown("""<a id="random-forest-testing-results"></a>""", unsafe_allow_html=True)
rf_testing_data = {
    'Parameter Type': ['mtry (max_features)', 'mtry (max_features)', 'mtry (max_features)', 'ntree (n_estimators)', 'ntree (n_estimators)', 'ntree (n_estimators)', 'Non-replacement RF'],
    'Parameter Value': [1, 2, 3, 100, 150, 200, 'N/A'],
    'Weighted RMSE': [22.7337, 22.5306, 22.5759, 22.5306, 22.6052, 22.6405, 25.9697],
    'Unweighted RMSE': [102.2232, 101.6853, 102.1483, 101.6853, 101.9057, 102.0619, 115.3067]
}
df_rf_testing = pd.DataFrame(rf_testing_data)
st.table(df_rf_testing)

st.markdown("""
The best Weighted RMSE is observed with `mtry = 2` and `ntree = 100` (22.5306).  
The best Unweighted RMSE is also achieved with `mtry = 2` and `ntree = 100` (101.6853).  
Non-replacement Random Forest shows higher RMSE values, indicating poorer performance compared to other configurations.  
Therefore, **mtry = 2** and **ntree = 100** are selected based on this testing.
""", unsafe_allow_html=True)

## Drop Testing Results
st.subheader("Random Forest Drop Testing Results")
st.markdown("""<a id="random-forest-drop-testing-results"></a>""", unsafe_allow_html=True)
rf_drop_data = {
    'Dropped Variable': ['None (Baseline)', 'DISTANCE', 'LARGE_AIRPORT', 'HAS_PASSENGERS', 'PASSENGERS', 'IS_WINTER'],
    'Weighted RMSE': [22.5306, 22.8714, 22.5250, 21.9451, 22.3390, 22.0356],
    'Difference from Baseline': ['0.0000', '+0.3408', '-0.0056', '-0.5855', '-0.1915', '-0.4950']
}
df_rf_drop = pd.DataFrame(rf_drop_data)
st.table(df_rf_drop)

st.markdown("""
Based on the drop testing, dropping `HAS_PASSENGERS` significantly improves performance, but logically, this variable makes sense as a predictor together with `PASSENGERS`. Therefore, we decided to keep it. Similarly, all other variables like `IS_WINTER` contribute meaningfully, and none should be dropped to maintain predictive accuracy and logical consistency.
""", unsafe_allow_html=True)

## Feature Importance
st.subheader("Feature Importance")
st.markdown("""<a id="feature-importance"></a>""", unsafe_allow_html=True)
feature_importance_data = {
    'Feature': ['DISTANCE', 'PASSENGERS', 'UNIQUE_CARRIER_Other', 'IS_WINTER', 'HAS_PASSENGERS', 'LARGE_AIRPORT', 'UNIQUE_CARRIER_Delta Air Lines Inc.', 'UNIQUE_CARRIER_SkyWest Airlines Inc.', 'UNIQUE_CARRIER_Federal Express Corporation', 'UNIQUE_CARRIER_Spirit Air Lines', 'UNIQUE_CARRIER_Endeavor Air Inc.', 'UNIQUE_CARRIER_United Air Lines Inc.', 'UNIQUE_CARRIER_ExpressJet Airlines LLC d/b/a aha!', 'UNIQUE_CARRIER_Southwest Airlines Co.', 'UNIQUE_CARRIER_Republic Airline', 'UNIQUE_CARRIER_United Parcel Service', 'UNIQUE_CARRIER_Envoy Air', 'UNIQUE_CARRIER_JetBlue Airways', 'UNIQUE_CARRIER_American Airlines Inc.', 'UNIQUE_CARRIER_GoJet Airlines LLC d/b/a United...', 'UNIQUE_CARRIER_Shuttle America Corp.', 'UNIQUE_CARRIER_PSA Airlines Inc.', 'UNIQUE_CARRIER_Compass Airlines', 'UNIQUE_CARRIER_Frontier Airlines Inc.', 'UNIQUE_CARRIER_Chautauqua Airlines Inc.'],
    'Importance': [0.540547, 0.341734, 0.049398, 0.019901, 0.017056, 0.015274, 0.005022, 0.002996, 0.001800, 0.001192, 0.000822, 0.000745, 0.000584, 0.000551, 0.000482, 0.000470, 0.000261, 0.000257, 0.000204, 0.000179, 0.000158, 0.000120, 0.000090, 0.000088, 0.000070]
}
df_feature_importance = pd.DataFrame(feature_importance_data)
st.table(df_feature_importance)

# Categorize features
df_feature_importance['Category'] = df_feature_importance['Feature'].apply(lambda x: 'UNIQUE_CARRIER' if 'UNIQUE_CARRIER' in x else x)

# Group by category and sum importance
bundled_importance = df_feature_importance.groupby('Category')['Importance'].sum().reset_index()
bundled_importance = bundled_importance.rename(columns={"Category": "Feature", "Importance": "Total Importance"})

# Sort the DataFrame
bundled_importance_sorted = bundled_importance.sort_values(by='Total Importance', ascending=False)

# Display the table in Streamlit
st.table(bundled_importance_sorted)

## Partial Dependence Plots
st.subheader("Partial Dependence Plots")
st.markdown("""<a id="partial-dependence-plots"></a>""", unsafe_allow_html=True)
st.markdown("""
**Partial Dependence Plot for DISTANCE**
""", unsafe_allow_html=True)
pd_distance_path = 'DATA/pd_distance.png'
if os.path.exists(pd_distance_path):
    st.image(pd_distance_path, caption='Partial Dependence Plot: DISTANCE')
else:
    st.warning('Partial Dependence Plot for DISTANCE not found.')

st.markdown("""
At lower `DISTANCE` values, the partial dependence remains consistent.  
Around 1750 miles, the prediction drops sharply, indicating a threshold where `DISTANCE` significantly reduces the predicted value, then recovers quickly. This indicates a subset of data with certain behavior.
""", unsafe_allow_html=True)

st.markdown("""
**Partial Dependence Plot for PASSENGERS**
""", unsafe_allow_html=True)
pd_passenger_path = 'DATA/pd_passenger.png'
if os.path.exists(pd_passenger_path):
    st.image(pd_passenger_path, caption='Partial Dependence Plot: PASSENGERS', use_column_width=True)
else:
    st.warning('Partial Dependence Plot for PASSENGERS not found.')

st.markdown("""
The partial dependence plots (PDPs) show how `PASSENGERS` influences `GROUND_TIME`, separated by the binary variable `HAS_PASSENGERS`.

`GROUND_TIME` generally increases as the number of `PASSENGERS` increases. There are fluctuations in the dependence (e.g., at 85 and 180 passengers), which might be due to variations in the data or interactions with other features.

The second plot shows no meaningful values for `PASSENGERS` because it doesn't make sense to have passengers (`PASSENGERS > 0`) when `HAS_PASSENGERS = 0`. This highlights the limitation of PDPs for certain predictors when logical relationships (e.g., `HAS_PASSENGERS = 0` implying `PASSENGERS = 0`) exist in the data.
""", unsafe_allow_html=True)

# Unique Carrier Analysis
st.subheader("Unique Carrier Analysis")
st.markdown("""<a id="unique-carrier-analysis"></a>""", unsafe_allow_html=True)
carrier_data = {
    '#': list(range(1, 21)),
    'Unique Carrier': [
        'Delta Air Lines Inc.', 'SkyWest Airlines Inc.', 'Endeavor Air Inc.', 'Spirit Air Lines', 'Other',
        'Frontier Airlines Inc.', 'United Air Lines Inc.', 'Southwest Airlines Co.', 'American Airlines Inc.',
        'Republic Airline', 'United Parcel Service', 'Federal Express Corporation', 'JetBlue Airways',
        'PSA Airlines Inc.', 'GoJet Airlines LLC d/b/a United Express', 'Envoy Air',
        'Chautauqua Airlines Inc.', 'Shuttle America Corp.', 'ExpressJet Airlines LLC d/b/a aha!', 'Compass Airlines'
    ],
    'Number of Flights': [4113, 1431, 1246, 1109, 553, 517, 507, 453, 316, 257, 152, 149, 123, 88, 41, 35, 0, 0, 0, 0],
    'RF Relative Partial Dependence': [
        1.806099, 2.213862, 1.943624, 1.799692, -12.849375, 2.500444, 2.375899, 0.938198, 3.000000, 2.647523,
        1.040208, 1.496569, 1.931597, 2.832963, 2.280391, 3.570444, 1.590978, 1.592344, 1.522513, 1.578054
    ],
    'GLM Relative Coefficient': [
        -1.136041, 2.913571, 1.701166, -1.948671, -0.482051, 0.026672, 1.309540, -5.357402, 3.000000, 5.440861,
        -3.759744, -0.898076, 0.063237, 6.809149, 2.614048, 8.699218, 2.927179, 1.278815, 0.566558, 2.516478
    ]
}
df_carrier_analysis = pd.DataFrame(carrier_data)
st.table(df_carrier_analysis)


######################### Sidebar Navigation ##################
