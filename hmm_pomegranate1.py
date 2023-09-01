import h5py
import numpy as np
import pomegranate as pomegranate
from pomegranate import Distribution, HiddenMarkovModel, DiscreteDistribution, State
import matplotlib.pyplot as plt


# def preprocess_spike_data_with_history(binned_spike_data, history_bins):
#      data_with_history = np.mean(binned_spike_data,axis=1)
#      data_with_history = (data_with_history-np.min(data_with_history))/(np.max(data_with_history)-np.min(data_with_history))
#      data_with_history = data_with_history.transpose().tolist()
# #     for i in range(history_bins, len(binned_spike_data)):
# #         data_with_history.append(binned_spike_data[i-history_bins:i+1].flatten().astype(int))
#      return data_with_history


class CustomHistoryEmissionDistribution(Distribution):
    def __init__(self, rates, history_bins, history_weight):
        self.rates = rates
        self.history_bins = history_bins
        self.history_weight = history_weight
        super().__init__()

    def probability(self, point):
        history_sum = np.sum(point[:-1] * self.history_weight)
        rate = self.rates[int(point[-1])]
        return np.exp(-rate) * (rate ** history_sum) / np.math.factorial(int(history_sum))

    def log_probability(self, point):
        return np.log(self.probability(point))


def compute_PUD_PDU(hidden_states):
    num_up_down = np.sum((hidden_states[:-1] == 1) & (hidden_states[1:] == 0))
    num_down_up = np.sum((hidden_states[:-1] == 0) & (hidden_states[1:] == 1))
    num_up = np.sum(hidden_states == 1)
    num_down = np.sum(hidden_states == 0)

    PUD = num_up_down / num_up if num_up != 0 else 0.0
    PDU = num_down_up / num_down if num_down != 0 else 0.0

    return PUD, PDU


def detect_UP_DOWN_phases(binned_spike_data, bin_size=25, history_bins=2, history_weight=0.01, transition_matrix=None, rates=np.array([0.5, -0.5]).reshape(-1, 1)):

    data_with_history = np.mean(binned_spike_data, axis=1)
    data_with_history = (data_with_history-np.min(data_with_history)) / \
                         (np.max(data_with_history)-np.min(data_with_history))
    data_with_history = data_with_history.transpose().tolist()

    if transition_matrix is None:
        print("Initializing transition matrix...")
        transition_matrix = np.array([[0.5, 0.5],
                                      [0.5, 0.5]])

    if rates is None:
        rates = np.array([1, -0.75]).reshape(-1, 1)

    model = HiddenMarkovModel()
     #Create a Poisson distribution object
#     poisson = pomegranate.PoissonDistribution()
#     model = HiddenMarkovModel.from_samples(
#     distribution=poisson,
#     n_components=2,
#     X=data_with_history,
#     tol=1e-6
# )

    up_distribution = CustomHistoryEmissionDistribution(
        rates[0], history_bins, history_weight)
    down_distribution = CustomHistoryEmissionDistribution(
        rates[1], history_bins, history_weight)

    up_state = State(up_distribution, name="UP")
    down_state = State(down_distribution, name="DOWN")

    model.add_states([up_state, down_state])
    model.add_transition(model.start, up_state, 1)
    model.add_transition(model.start, down_state, 0)
    model.add_transition(up_state, up_state, 0.9)
    model.add_transition(up_state, down_state, 0.1)
    model.add_transition(down_state, down_state, 0.9)
    model.add_transition(down_state, up_state, 0.1)

    model.bake([data_with_history])
    print('baking done')
    model.fit([data_with_history], stop_threshold=1e-5, verbose=True)
    print('Fitting done...')
   
    hidden_states = model.predict(data_with_history)
    PUD, PDU = compute_PUD_PDU(hidden_states)
    P_UP = model.predict_proba(data_with_history)[:, 0] # get probability of being in UP state
    
    return hidden_states, PUD, PDU, P_UP, data_with_history

filename = 'C:\\Users\\AD263755\\Documents\\MATLAB\\UpDownStatesCodes\\myHmmAD\\datapy.mat'

with h5py.File(filename, 'r') as file:
    # List all variables in the .mat file
    print("Variables in the .mat file:", list(file.keys()))

    # Load the variables you need
    rates = np.array(file['rates'])
    spikeCounts = np.array(file['spikeCounts'])
    # spikeCounts = spikeCounts.T
    
# select first 20 seconds
start_time = 1
end_time = 250 * 20
time = np.arange(start_time, end_time) / 20

spikeCounts = spikeCounts[start_time:end_time+1,:]
trans_mat = np.array([[0.5, 0.5], [0.5, 0.5]])
em_mat = np.array([[0.9], [-0.5]])
hidden_states, PUD, PDU, P_UP, data = detect_UP_DOWN_phases(spikeCounts,bin_size=5, history_bins=2, history_weight=0.01, transition_matrix=trans_mat, rates=em_mat)

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

# show plot
plt.show()
    
    

