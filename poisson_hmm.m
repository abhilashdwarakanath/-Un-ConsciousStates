function [state_sequence, final_state_sequence] = poisson_hmm(data,~)

% % Downsample data
% 
% data = data(1:10:end);
% t = t(1:10:end);

% Perform clustering
k = 2; % Number of clusters

[cluster_indices] = kmeans(data', k, 'Replicates', 10);

init_down_state = data(cluster_indices==1);
init_up_state = data(cluster_indices==2);

lambda1 = mean(init_down_state);
lambda2 = mean(init_up_state);

lambda = [lambda1 lambda2]./(lambda1+lambda2);

% Create state sequence
state_sequence = smoothOverChunks(cluster_indices,0.1,20);

% Train Poisson HMM
%TRANS_GUESS = 0.5+zeros(k);

TRANS_GUESS = [lambda(1) 1-lambda(1); 1-lambda(2) lambda(2)];

%EMIS_GUESS = poisspdf(repmat(1:T, k, 1), repmat([lambda(1); lambda(2)], 1, T));

%EMIS_GUESS = zeros(2,length(t));

% EMIS_GUESS(1,1) = 1-lambda(1);
% EMIS_GUESS(1,2) = lambda(1);
% EMIS_GUESS(2,1) = lambda(2);
% EMIS_GUESS(2,2) = 1-lambda(2);

EMIS_GUESS = 0.5+zeros(k);

[TRANS_EST, EMIS_EST] = hmmtrain(state_sequence, TRANS_GUESS, EMIS_GUESS,'Algorithm', 'BaumWelch', 'Maxiterations', 1e5,'Tol',1e-6,'Verbose',true);

% Compute the final state sequence using hmmviterbi
final_state_sequence = hmmviterbi(state_sequence, TRANS_EST, EMIS_EST);