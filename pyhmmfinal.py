# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:21:35 2023

@author: Abhilash Dwarakanath
"""

import h5py
import numpy as np
from pomegranate import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def detect_UP_DOWN_phases(binned_spike_data, t00, t01, t10, t11):

    input_data = np.mean(binned_spike_data, axis=1)
    
    # do k-means to initialise emission rates
    kmeans = KMeans(n_clusters=2, random_state=0, tol=1e-6, n_init=10, verbose=True)
    
    # Fit the KMeans model to the data
    kmeans.fit(input_data.reshape(-1,1))
    print('Clustering done...')
    # Get the centroids of the clusters
    cluster_means = kmeans.cluster_centers_

    # Flatten the cluster means to a 1D array
    #rates = cluster_means.ravel()
    rates = list(cluster_means.ravel())
    print('Flattening done...')

    model = HiddenMarkovModel()
    print('Model defined...')
    
    up_state = State(PoissonDistribution(float(rates[0])), name="UP")
    down_state = State(PoissonDistribution(float(rates[1])), name="DOWN")

    print('State probabilities distributions defined...')

    print('Before adding states...')
    model.add_states(list([up_state, down_state]))
    print('States added successfully...')

    print('Before adding start transitions...')
    model.add_transition(model.start, up_state, 0.5) 
    model.add_transition(model.start, down_state, 0.5)
    print('Start transitions added successfully...')

    print('Before adding state transitions...')
    # Use individual transition probabilities in the add_transition() function calls
    model.add_transition(up_state, up_state, t00)
    model.add_transition(up_state, down_state, t01)
    model.add_transition(down_state, down_state, t10)
    model.add_transition(down_state, up_state, t11)
    print('State transitions added successfully...')

    model.bake()
    print('baking done')
    model.fit([input_data], verbose=True, stop_threshold=1e-6)
    print('Fitting done...')
   
    hidden_states = model.predict(input_data)
    print('Successfully predicted hidden state sequence...')
    P_UP = model.predict_proba(input_data)[:, 1] # get probability of being in UP state
    print('Probability of State 1 predicted...')
    
    # Estimate the transition probabilities from the posterior probabilities
    posterior_probs = model.predict_proba(input_data)
    print('Posterior probabilities predicted...')
    posterior_transition_probs = np.zeros((2, 2))
    for i in range(len(input_data) - 1):
        posterior_transition_probs[hidden_states[i+1], hidden_states[i]] += posterior_probs[i+1, hidden_states[i+1]]
    posterior_transition_probs /= np.sum(posterior_probs[:-1], axis=0)
    print('Final transition probabilities predicted...')
    return hidden_states, P_UP, input_data, posterior_transition_probs

# filename = 'C:\\Users\\Abhilash Dwarakanath\\Documents\\MATLAB\\UpDownStatesCodes\\datapy.mat'

# with h5py.File(filename, 'r') as file:
#     # List all variables in the .mat file
#     print("Variables in the .mat file:", list(file.keys()))

#     # Load the variables you need
#     rates = np.array(file['rates'])
#     spikeCounts = np.array(file['spikeCounts'])
#     # spikeCounts = spikeCounts.T
    
# # select first 20 seconds
# start_time = 1
# end_time = 500 * 20
# time = np.arange(start_time, end_time) / 20

# spikeCounts = spikeCounts[start_time:end_time+1,:]

# t00 = 0.9
# t01 = 0.1
# t10 = 0.1
# t11 = 0.9
# hidden_states, P_UP, data, fitted_tr_mat = detect_UP_DOWN_phases(spikeCounts, t00, t01, t10, t11)

# # calculate mean of spikeCounts
# mean_spikeCounts = spikeCounts.mean(axis=1)

# # create figure and axes
# fig, ax1 = plt.subplots()

# # plot mean_spikeCounts on first y-axis
# ax1.plot(time, mean_spikeCounts[start_time:end_time], color='blue')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Mean Spike Count', color='blue')

# # create second y-axis and plot P_UP
# ax2 = ax1.twinx()
# ax2.plot(time, P_UP[start_time:end_time], color='red')
# ax2.set_ylabel('P_UP', color='red')
# plt.show()