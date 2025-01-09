# This script calculates the Exponentially Weighted Moving Average (EWMA) for time series data.
# The function `ewma` computes the EWMA by applying the formula:
# EWMA(t) = alpha * Data(t) + (1 - alpha) * EWMA(t-1)
# where `alpha` is a smoothing factor that controls the weight given to recent data points.
# The script does the following:
# 1. Defines the `ewma` function to compute the EWMA for a given dataset.
# 2. Uses a sample dataset (time series data) and applies the `ewma` function.
# 3. Prints the original data and the resulting EWMA values.
# 4. Plots the original data and EWMA values in two ways:
#    - A 2D line plot showing the original data and the smoothed EWMA series.
#    - A 3D scatter plot to visualize the data and EWMA over time.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ewma(data, alpha):
    ewma_values = [data[0]]
    for i in range(1, len(data)):
        ewma_value = alpha * data[i] + (1 - alpha) * ewma_values[i - 1]
        ewma_values.append(ewma_value)
    return np.array(ewma_values)

data = np.array([10, 20, 15, 30, 25, 35, 30, 40, 45, 50])
alpha = 0.8
ewma_result = ewma(data, alpha)

print("Original Data:", data)
print("EWMA Result:", ewma_result)

plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data", marker='o', linestyle='--', color='blue')
plt.plot(ewma_result, label="EWMA", marker='x', color='red', linewidth=2)
plt.title('Exponentially Weighted Moving Average (EWMA) - 2D')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
time_index = np.arange(len(data))
ax.scatter(time_index, data, zs=0, zdir='z', label='Original Data', color='blue', marker='o')
ax.scatter(time_index, ewma_result, zs=1, zdir='z', label='EWMA', color='red', marker='x')
ax.set_title('Exponentially Weighted Moving Average (EWMA) - 3D')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_zlabel('EWMA')
ax.legend()
plt.show()

