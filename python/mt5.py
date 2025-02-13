import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
# Initialize MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MetaTrader 5")
    quit()

# Define parameters
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_D1  # 1-hour timeframe
years = 5  # Number of years to fetch

# Calculate the start and end time for data
# end_time = datetime.now()
# start_time = end_time - timedelta(days=years * 365)  # Approximate days in years
start_time = datetime(2018, 11, 1) - timedelta(days=89) #(YYYY, MM, DD, HH, MM, SS)
end_time = datetime(2025, 2, 11)   #(YYYY, MM, DD, HH, MM, SS)

# Fetch historical data
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
if rates is None:
    print(f"Failed to fetch data for {symbol}")
    mt5.shutdown()
    quit()

# Convert data to Pandas DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')  # Convert timestamp to datetime

# Calculate EMA with a window size of 89
ema_window = 89
data['ema_89'] = data['close'].ewm(span=ema_window, adjust=False).mean()
#
# 
#  Convert 'time' to datetime
timestamp = pd.to_datetime(data['time'])

# Calculate seconds since midnight
seconds_since_midnight = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second

# Normalize time to [0, 1]
normalized_time = seconds_since_midnight / (24 * 3600)

# Calculate sinusoidal features
data['time_sin'] = np.sin(2 * np.pi * normalized_time)
data['time_cos'] = np.cos(2 * np.pi * normalized_time)

# Print and store the updated DataFrame
print(data)

# Shutdown MetaTrader 5
mt5.shutdown()


# Calculate price difference between last two candles
data['price_diff'] = data['close'].diff()



def calculate_trendline(data):
    x = np.array(range(len(data)))
    y = data['close'].values
    coefficients = np.polyfit(x, y, 1)
    return coefficients[0], coefficients[1], x  # Slope and intercept



window_size = 89
import matplotlib.pyplot as plt



for i in range(len(data) - window_size + 1):
   
    window_data = data.iloc[i:i+window_size]
    slope, intercept, x = calculate_trendline(window_data)
    trendline = slope * x + intercept  # Line equation: y = mx + b
    data.loc[i + window_size - 1, 'trendline'] = trendline[-1]
  

# Convert datetime to a string without invalid characters
start_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")  # Replace : with -
end_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")      # Replace : with -
# Save to a CSV file
data.to_csv(f"{symbol}_{start_str}_{end_str}_{timeframe}.csv", index=False)

# Display the first few rows
print(data.head())