import pandas as pd
import numpy as np


# Read the CSV file
data = pd.read_csv("./output.csv")

# Select relevant columns
df = data[['time', 'close', 'tick_volume', 'price_diff', 'trendline']]

# Convert 'time' to datetime
timestamp = pd.to_datetime(df['time'])

# Calculate seconds since midnight
seconds_since_midnight = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second

# Normalize time to [0, 1]
normalized_time = seconds_since_midnight / (24 * 3600)

# Calculate sinusoidal features
df['time_sin'] = np.sin(2 * np.pi * normalized_time)
df['time_cos'] = np.cos(2 * np.pi * normalized_time)

# Print and store the updated DataFrame
print(df)

# Save the updated DataFrame to a new CSV file
df.to_csv("updated_output.csv", index=False)