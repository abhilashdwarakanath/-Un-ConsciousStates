import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution, State

def bin_spike_data(spike_data):
    return np.sum(spike_data, axis=0).reshape(-1, 1)

def preprocess_spike_data_with_history(binned_spike_data, history_bins):
    data_with_history = []
    for i in range(history_bins, len(binned_spike_data)):
        data_with_history.append(binned_spike_data[i-history_bins:i+1].flatten())
    return np.array(data_with_history)

class CustomHistoryEmissionDistribution(DiscreteDistribution):
    def __init__(self, rates, history_bins, history_weight):
        self.rates = rates
        self.history_bins = history_bins
        self.history_weight = history_weight
        super().__init__({})

    def probability(self, point):
        history_sum = np.sum(point[:-1] * self.history_weight)
        rate = self.rates[int(point[-1])]
        return np.exp(-rate) * (rate ** history_sum) / np.math.factorial(int(history_sum))

def detect_UP_DOWN_phases(spike_data, bin_size=10, history_bins=2, history_weight=0.01, transition_matrix=None, rates=None):
    binned_spike_data = bin_spike_data(spike_data)
    data_with_history = preprocess_spike_data_with_history(binned_spike_data, history_bins)
    
    if transition_matrix is None:
        transition_matrix = np.array([[0.1, 0.9],
                                      [0.9, 0.1]])
    
    if rates is None:
        rates = np.array([3, -2]).reshape(-1, 1)
    
    model = HiddenMarkovModel()
    
    up_distribution = CustomHistoryEmissionDistribution(rates, history_bins, history_weight)
    down_distribution = CustomHistoryEmissionDistribution(-rates, history_bins, history_weight)
    
    up_state = State(up_distribution, name="UP")
    down_state = State(down_distribution, name="DOWN")
    
    model.add_states([up_state, down_state])
    model.add_transition(model.start, up_state, 1)
    model.add_transition(model.start, down_state, 0)

    for i in range(2):
        for j in range(2):
            model.add_transition(model.states[i], model.states[j], transition_matrix[i, j])
    
    model.bake()
    model.fit(data_with_history)
    hidden_states = model.predict(data_with_history)
    
    return hidden_states