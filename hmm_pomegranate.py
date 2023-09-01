import h5py
import numpy as np
from pomegranate import *
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt

def detect_UP_DOWN_phases(binned_spike_data, transition_matrix, history_weight):

    init_sequence = np.mean(binned_spike_data, axis=1)
    transition_matrix = transition_matrix * history_weight;
    
    # do k-means to initialise emission rates
    kmeans = KMeans(n_clusters=2, random_state=0, tol=1e-6, n_init=10, verbose=True)

    # Fit the KMeans model to the data
    kmeans.fit(init_sequence.reshape(-1,1))
    
    # Get the centroids of the clusters
    cluster_means = kmeans.cluster_centers_

    # Flatten the cluster means to a 1D array
    rates = cluster_means.ravel()
    #fr_mean = np.mean(fr)
    #init_sequence = np.where(fr > fr_mean,2,1)

    if transition_matrix is None:
        print("Initializing transition matrix...")
        transition_matrix = np.array([[0.5, 0.5],
                                      [0.5, 0.5]])

    model = HiddenMarkovModel()

    up_state = State(PoissonDistribution(rates[0]), name="UP")
    down_state = State(PoissonDistribution(rates[1]), name="DOWN")

    model.add_states([up_state, down_state])
    model.add_transition(model.start, up_state, 0.5) 
    model.add_transition(model.start, down_state, 0.5) 
    model.add_transition(up_state, up_state, transition_matrix[0, 0])
    model.add_transition(up_state, down_state, transition_matrix[0, 1])
    model.add_transition(down_state, down_state, transition_matrix[1, 0])
    model.add_transition(down_state, up_state, transition_matrix[1, 1])

    model.bake()
    print('baking done')
    model.fit([init_sequence], verbose=True, stop_threshold=1e-6)
    print('Fitting done...')
   
    hidden_states = model.predict(init_sequence)
    P_UP = model.predict_proba(init_sequence)[:, 0] # get probability of being in UP state
    
    # Estimate the transition probabilities from the posterior probabilities
    posterior_probs = model.predict_proba(init_sequence)
    posterior_transition_probs = np.zeros((2, 2))
    for i in range(len(init_sequence) - 1):
        posterior_transition_probs[hidden_states[i+1]-1, hidden_states[i]-1] += posterior_probs[i+1, hidden_states[i+1]-1]
    posterior_transition_probs /= np.sum(posterior_probs[:-1], axis=0)
    
    return hidden_states, P_UP, init_sequence, posterior_transition_probs


filename = 'C:\\Users\\Abhilash Dwarakanath\\Documents\\MATLAB\\UpDownStatesCodes\\datapy.mat'

with h5py.File(filename, 'r') as file:
    # List all variables in the .mat file
    print("Variables in the .mat file:", list(file.keys()))

    # Load the variables you need
    rates = np.array(file['rates'])
    spikeCounts = np.array(file['spikeCounts'])
    # spikeCounts = spikeCounts.T
    
# select first 20 seconds
start_time = 1
end_time = 500 * 20
time = np.arange(start_time, end_time) / 20

spikeCounts = spikeCounts[start_time:end_time+1,:]

trans_mat = np.array([[0.5, 0.5], [0.5, 0.5]])
hidden_states, P_UP, data, fitted_tr_mat = detect_UP_DOWN_phases(spikeCounts,transition_matrix=trans_mat,history_weight=0.1)

# calculate mean of spikeCounts
mean_spikeCounts = spikeCounts.mean(axis=1)

# create figure and axes
fig, ax1 = plt.subplots()

# plot mean_spikeCounts on first y-axis
ax1.plot(time, mean_spikeCounts[start_time:end_time], color='blue')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Mean Spike Count', color='blue')

# create second y-axis and plot P_UP
ax2 = ax1.twinx()
ax2.plot(time, P_UP[start_time:end_time], color='red')
ax2.set_ylabel('P_UP', color='red')
plt.show()

# save the figure as a pickle file

with open('hmmTest.pickle', 'wb') as f:
    pickle.dump(fig, f)

