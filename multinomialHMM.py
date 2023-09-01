# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:08:07 2023

@author: Abhilash Dwarakanath
"""

import h5py
import numpy as np
from scipy.stats import poisson
from hmmlearn import hmm
import matplotlib.pyplot as plt

filename = 'C:\\Users\\Abhilash Dwarakanath\\Documents\\MATLAB\\UpDownStatesCodes\\datapy.mat'

with h5py.File(filename, 'r') as file:
    # List all variables in the .mat file
    print("Variables in the .mat file:", list(file.keys()))

    # Load the variables you need
    rates = np.array(file['rates'])
    spikeCounts = np.array(file['spikeCounts'])
    # spikeCounts = spikeCounts.T

# calculate mean of spikeCounts
data = spikeCounts.mean(axis=1)


# Calculate first and last quartiles of the data
q1_data = np.percentile(data, 25)
q3_data = np.percentile(data, 75)

# Determine the maximum observed value in the dataset
max_value = int(np.max(data))

# Generate Poisson distributions for each state using quartiles as rate (lambda) parameters
state1_poisson = poisson.pmf(np.arange(max_value + 1), q1_data)
state2_poisson = poisson.pmf(np.arange(max_value + 1), q3_data)

# Combine state Poisson distributions into the emission matrix (2 x M)
emission_matrix = np.vstack((state1_poisson, state2_poisson))

# Initialize and fit the HMM model
model = hmm.MultinomialHMM(n_components=2, tol=1e-6)
model.emissionprob_ = emission_matrix
model = model.fit(observations)

# Predict hidden state sequence
hidden_states = model.predict(observations)

# Calculate log-likelihood
log_likelihood = model.score(observations)

# Compute the P(state) path for each state (posterior probabilities)
posteriors = model.predict_proba(observations)

print("Hidden state sequence:", hidden_states)
print("Log-likelihood:", log_likelihood)
print("P(state) path for each state:")
print(posteriors)

time = np.arange(1, end_time) / 20

# create figure and axes
fig, ax1 = plt.subplots()

# plot mean_spikeCounts on first y-axis
ax1.plot(time, mean_spikeCounts, color='blue')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Mean Spike Count', color='blue')

# create second y-axis and plot P_UP
ax2 = ax1.twinx()
ax2.plot(time, P_UP, color='red')
ax2.set_ylabel('P_UP', color='red')
plt.show()
