function [stateSeq, transProb, emisProb,counts,bins] = poissonHMM(data, nStates,t)

% Fits 2 n-state Poisson HMM to the given spike time data.
% AD. NS. 2023

% Convert spike times to ms for greater precision in spike timing

tStart = t(1)*1e3;
tEnd = t(end)*1e3;

spikeTimes = data*1e3;

% Bin spikes

bins = tStart:50:tEnd; % bin spikes in 50ms bins

counts = histc(spikeTimes,bins);
zs = zscore(counts);

% Initialize the model parameters via poisson fit for emisProb
transProbGuess = 0.5+zeros(nStates);
l = poissfit(counts);

emisProbGuess(1,1) = l; emisProbGuess(1,2) = 1-l;
emisProbGuess(2,1) = 1-l; emisProbGuess(2,2) = l;

% Create initial sequence guess
seq = ones(1,length(counts));
seq(zs>0) = 2;

% Run the Baum-Welch algorithm to fit the HMM
[transProb, emisProb] = hmmtrain(seq, transProbGuess, emisProbGuess,'Tolerance', 1e-6,'Maxiterations',1e6,'Verbose',true);

% Compute the most likely state sequence using the Viterbi algorithm
stateSeq = hmmviterbi(state_sequence, transProb, emisProb);
end