# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:40:06 2023

@author: chris
"""

# %% Volatility
import numpy as np
import matplotlib.pyplot as plt
import math

# Generate 100,000 random draws
prices = np.cumsum(np.random.normal(0, 0.01, 10000))  # Cumulative sum for price simulation

# Calculate logarithmic returns
log_returns = np.log(prices[1:] / prices[:-1])

# Calculate historical volatility with a rolling window of 30 observations
window_size = 30
volatility = np.zeros_like(prices)
volatility[:window_size] = np.std(log_returns[:window_size])

for i in range(window_size, len(prices)):
    volatility[i] = np.std(log_returns[i - window_size + 1:i + 1])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Price')
plt.plot(volatility, label='Volatility (Rolling Window)')
plt.legend()
plt.title('Price and Historical Volatility')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.show()