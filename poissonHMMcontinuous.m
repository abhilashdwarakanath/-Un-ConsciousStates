function [stateSeq, transProb, emisProb] = poissonHMMcontinuous(data, nStates)

% Fits 2 n-state Poisson HMM to the given spike time data.
% AD. NS. 2023
data = smooth(data);
data = normalise(data.^2);

% Initialize the model parameters via poisson fit for emisProb
transProbGuess = 0.5+zeros(nStates);
l = poissfit(data);

emisProbGuess(1,1) = l; emisProbGuess(1,2) = 1-l;
emisProbGuess(2,1) = 1-l; emisProbGuess(2,2) = l;

% Create initial sequence guess
seq = ones(1,length(data));
seq(data>median(data)) = 2;

% Run the Baum-Welch algorithm to fit the HMM
[transProb, emisProb] = hmmtrain(seq, transProbGuess, emisProbGuess,'Tolerance', 1e-6,'Maxiterations',1e6,'Verbose',true);

% Compute the most likely state sequence using the Viterbi algorithm
stateSeq = hmmviterbi(seq, transProb, emisProb);
end