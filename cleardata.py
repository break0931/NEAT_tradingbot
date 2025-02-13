import pandas as pd
import numpy as np

# Read the CSV file
data = pd.read_csv("XAUUSD_M15_5years.csv",sep='\t')
data.columns = [col.strip("<>") for col in data.columns]
# Ensure column names are used correctly
data['price_diff'] = data['CLOSE'].diff()

# Function to calculate trendline (slope, intercept)
def calculate_trendline(data):
    x = np.arange(len(data))  # Generate index array
    y = data['CLOSE'].values  # Use correct column name
    coefficients = np.polyfit(x, y, 1)  # Fit linear regression
    return coefficients[0], coefficients[1]

window_size = 89
trendline_values = [np.nan] * (window_size - 1)  # Fill first (window_size - 1) with NaN

# Calculate trendline for each window
for i in range(len(data) - window_size + 1):
    window_data = data.iloc[i : i + window_size]
    slope, intercept = calculate_trendline(window_data)
    trendline_values.append(slope * (window_size - 1) + intercept)  # Store last trendline point

# Append trendline values to DataFrame
data['trendline'] = trendline_values
data = data.rename(columns={"TICKVOL": "tick_volume"})
data = data.rename(columns={"CLOSE": "close"})

data['time'] = pd.to_datetime(data['DATE'] + " " + data['TIME'])
timestamp = pd.to_datetime(data['time'])
# Calculate seconds since midnight
seconds_since_midnight = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second

# Normalize time to [0, 1]
normalized_time = seconds_since_midnight / (24 * 3600)

# Calculate sinusoidal features
data['time_sin'] = np.sin(2 * np.pi * normalized_time)
data['time_cos'] = np.cos(2 * np.pi * normalized_time)

# Save to CSV
data.to_csv("XAUUSD_M15_5years_final.csv", index=False)

# Display first few rows
print(data.head())
