#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastdtw import fastdtw
import numpy as np

# Two example time series
time_series_a = np.array([1, 3, 4, 6, 9])
time_series_b = np.array([0, 2, 3, 5, 6, 8, 10])

# Compute the DTW distance using fastdtw
distance, path = fastdtw(time_series_a, time_series_b)

print("DTW Distance:", distance)
print("Optimal Path:", path)


# In[2]:


import numpy as np

def dtw_distance(x, y):
    # Calculate the pairwise distance between elements of x and y using the Euclidean distance
    distance_matrix = np.abs(np.subtract.outer(x, y))

    # Initialize the cost matrix with zeros
    cost_matrix = np.zeros((len(x), len(y)))

    # Initialize the first row and first column of the cost matrix
    cost_matrix[0, 0] = distance_matrix[0, 0]
    for i in range(1, len(x)):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + distance_matrix[i, 0]
    for j in range(1, len(y)):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + distance_matrix[0, j]

    # Fill in the rest of the cost matrix using dynamic programming
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            cost_matrix[i, j] = distance_matrix[i, j] + min(cost_matrix[i - 1, j],
                                                           cost_matrix[i, j - 1],
                                                           cost_matrix[i - 1, j - 1])

    # The DTW distance is the value in the bottom-right cell of the cost matrix
    dtw_distance = cost_matrix[-1, -1]

    return dtw_distance, cost_matrix

# Two example time series
time_series_a = np.array([1, 3, 4, 6, 9])
time_series_b = np.array([0, 2, 3, 5, 6, 8, 10])

# Compute the DTW distance using the custom implementation
distance, cost_matrix = dtw_distance(time_series_a, time_series_b)

print("DTW Distance:", distance)
print("Cost Matrix:")
print(cost_matrix)


# In[ ]:




