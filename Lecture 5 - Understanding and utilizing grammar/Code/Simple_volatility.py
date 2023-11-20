# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:40:06 2023

@author: chris
"""

# %% Volatility
import numpy as np
import matplotlib.pyplot as plt

# Generate 100,000 random draws
random_sequence = np.random.normal(0, 1, 100000)

# Calculate volatility using a rolling window of 100 observations
window_size = 100
volatility = np.zeros_like(random_sequence)

for i in range(window_size, len(random_sequence)):
    window = random_sequence[i - window_size: i]
    volatility[i] = np.std(window)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(random_sequence, label='Random Sequence')
plt.plot(volatility, label='Volatility (Rolling Window)')
plt.legend()
plt.title('Random Sequence and Volatility')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.show()