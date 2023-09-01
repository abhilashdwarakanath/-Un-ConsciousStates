# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:42:08 2023

@author: Abhilash Dwarakanath
"""

import numpy as np
from hmmlearn import hmm
import h5py

class PoissonSemiHMMWithHistory(hmm.PoissonHMM):
    def __init__(self, n_states=2, n_symbols=2, *args, **kwargs):
        super().__init__(n_components=n_states, *args, **kwargs)
        self.n_symbols = n_symbols

    def preprocess_data(self, data, history):
        preprocessed_data = np.zeros((len(data) - history, data.shape[1] * history))
        for i in range(len(data) - history):
            preprocessed_data[i, :] = data[i:i+history, :].flatten()
        return preprocessed_data

    def get_results(self, data):
        hidden_states = self.predict(data)
        state_probs = self.predict_proba(data)
        transition_matrix = self.transmat_
        return hidden_states, state_probs, transition_matrix



filename = 'C:\\Users\\Abhilash Dwarakanath\\Documents\\MATLAB\\UpDownStatesCodes\\datapy.mat'

with h5py.File(filename, 'r') as file:
    # List all variables in the .mat file
    print("Variables in the .mat file:", list(file.keys()))

    # Load the variables you need
    rates = np.array(file['rates'])
    spikeCounts = np.array(file['spikeCounts'])
    
# Usage example
n_neurons = 25
n_bins = 11549
n_states = 2
n_symbols = 2
history = 5


model = PoissonSemiHMMWithHistory(n_states=n_states, n_symbols=n_symbols)
preprocessed_data = model.preprocess_data(spikeCounts, history)
model.fit(preprocessed_data)

hidden_states, state_probs, transition_matrix = model.get_results(preprocessed_data)

print("Hidden states:", hidden_states)
print("Most likely path P(State 1):", state_probs)
print("Transition matrix:", transition_matrix)