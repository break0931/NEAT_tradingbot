import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Sample Data: Simulating High and Low Prices
data = {'date': pd.date_range(start='2024-01-01', periods=20),
        'high': [100, 145, 145, 110, 120, 115, 130, 125, 140, 135, 145, 140, 150, 155, 150, 160, 165, 160, 170, 175],
        'low': [90, 85, 88, 95, 100, 105, 110, 108, 115, 118, 120, 123, 125, 130, 128, 135, 138, 137, 140, 145]}
df = pd.DataFrame(data)

# Step 1: Identify Swing Highs and Lows
highs = df['high'].values
lows = df['low'].values
dates = df['date']

# Find peaks (swing highs)
high_indices, _ = find_peaks(highs, distance=10)  # Ensure spacing
swing_highs = df.iloc[high_indices]

# Find troughs (swing lows)
low_indices, _ = find_peaks(-lows, distance=10)  # Invert lows to find valleys
swing_lows = df.iloc[low_indices]

# Step 2: Fit Trendlines to Highs and Lows
x_high = np.arange(len(swing_highs))
y_high = swing_highs['high'].values
m_high, b_high = np.polyfit(x_high, y_high, 1)

x_low = np.arange(len(swing_lows))
y_low = swing_lows['low'].values
m_low, b_low = np.polyfit(x_low, y_low, 1)

# Step 3: Extend the Trendlines
df['trend_high'] = m_high * np.arange(len(df)) + b_high
df['trend_low'] = m_low * np.arange(len(df)) + b_low

# Plotting
plt.figure(figsize=(12, 6))

# Plot Highs, Lows, and Trendlines
plt.plot(df['date'], df['high'], label='High Prices', marker='o', linestyle='-')
plt.plot(df['date'], df['low'], label='Low Prices', marker='o', linestyle='-')
plt.scatter(swing_highs['date'], swing_highs['high'], color='red', label='Swing Highs', zorder=3)
plt.scatter(swing_lows['date'], swing_lows['low'], color='green', label='Swing Lows', zorder=3)
plt.plot(df['date'], df['trend_high'], linestyle="dashed", color='blue', label='High-to-High Trendline')
plt.plot(df['date'], df['trend_low'], linestyle="dashed", color='purple', label='Low-to-Low Trendline')

plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
