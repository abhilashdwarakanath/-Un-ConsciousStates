clear all
clc
close all

% Toy to play with simulated UP/DOWN states. AD. NS. 2023

dbstop if error

%% Establish parameters

% For simulating data. The durations of the UP and DOWN states are drawn from a Gamma distribution with fixed shape parameter, but the rates (mean) are defined
% below. Also defined below are - A range for the firing rates of neurons in UP state, a range for the firing rates of neurons in the DOWN state, the duration
% of the data you want to generate, and the number of neurons

params.Fs = 1000; %Hz
params.upFR = [15 35]; % spikes/s (Hz)
params.downFR = [0.01 2.5]; % spikes/s (Hz)
params.jitter = 0.25; % MEAN Seconds to jitter each neuron by so that they don't artifically start firing all at the same time. The jitter value is drawn from a Gaussian distribution
params.duration = 100; %seconds
params.meanDownDur = 0.86; %seconds
params.meanUpDur = 0.34; %seconds
params.nNeurons = 25; % variable ka naam dekhlo chutiyon

% After the synthetic data is generated, it needs to be processed into a population firing rate. The params below control that.

params.binWidth = 0.05; % seconds. Increase this for a smoother PSTH. You will lose fine-grained details. Reduce it too much and it gets unnecessarily noisy. How does one choose an appropriate bin width? Hahahahaha. *cries in electrophysiology*
params.binnedFs = 1/params.binWidth;
params.chunkSize = 100; % For display only. Rasters are too heavy. Do me a favour and don't set this below duration.

% Parameters for the HMM. The idea is that there are 2 states, one depolarising state that causes the neurons to fire, and one baseline phase which makes
% them silent. UP/DOWN states are commonly observed during Sleep and Anaesthesia, where neuronal populations increase their coherence locally. So we assume that
% all these neurons are being affected commonly by the two states. Therefore, they emit 2 symbols, one for firing above baseline, one for firing at or below
% baseline. 

params.nStates = 2;
params.nIter = 1000; % Maxmimum number of iterations for the B-W algorithm to run
params.tol = 1e-6; % Convergence criterion for fitting.
params.history = 0.1; % in seconds. This is to smooth over noisy bursts. In literature, values between 50-100ms are used. We will use the lower-bound so as to not smooth too much. In general, this should be set to 2 bin widths. Why? That's what I've seen works "best". No further questions.
params.holdOut = 0.2; % Proportion of data to hold-out for testing
params.nFolds = 10; % Number of folds to randomly split training data to train on for more robustness

%% Simulate UP/DOWN spike rasters

spikeTrains = simulateUPDOWNStatesGammaPoisson(params);

%% Process the data into PSTH and training folds

% Create the time vector
t = linspace(0,size(spikeTrains,2)./params.Fs,size(spikeTrains,2));

% Create the binning vector
bins = 0:params.binWidth:t(end);

% Initialise the matrix to populate with spike counts, and the spike times cell array(trust me, using a cell array to plot the raster is much better than
% imagesc() ). 
spikeCounts = zeros(size(spikeTrains,1),length(bins));
spikeTimes = cell(params.nNeurons,1);

% Loop through the channels and populate the spike raster. Also collect the spike times separately. These will be used to plot the raster plot
for iChan = 1:size(spikeTrains,1)
    idx = find(spikeTrains(iChan,:)==1);
    st = t(idx);
    spikeTimes{iChan,1} = st;
    spikeCounts(iChan,:) = histc(st,bins)./params.binWidth; % Divide by bin width to already get your PSTH in spikes/s
end

instFiringRate = nanmean(spikeCounts,1); % PSTH

% Normalise. Normalising or centering is important. If you're normalising between 0 and 1, make sure to cosntrain your vector between 1e-3 and 0.999 to avoid
% division by 0 or log(1) that will cause computers to whine.
instFiringRate = normalise(instFiringRate);
instFiringRate(instFiringRate==0) = 1e-3;
instFiringRate(instFiringRate==1) = 0.999;

% Split the data intro training and testing
nTrainingSamps = floor((1-params.holdOut)*length(instFiringRate));
trainingData = instFiringRate(1:nTrainingSamps);
testingData = instFiringRate(nTrainingSamps+1:end);

% Split the training data into folds

foldSize = floor(length(trainingData)/params.nFolds);
remainder = mod(length(trainingData), params.nFolds);
trainingData = trainingData(1:end-remainder);

foldedData = reshape(trainingData, length(trainingData)/params.nFolds, params.nFolds);

%% Start the fitting.

% Initialise the transition matrix. The transition matrix is an nStates x nStates matrix that holds the probabilities for each state and the transition
% probabilities from one state to another
initTransMat = 0.3+(0.7-0.3)*rand(params.nStates);

% initialise the transition and emission probability matrices to hold the fitted values.The emission matrix is an nStates x nSymbols matrix that holds the 
% probability of a particular symbol being emitted by a particular state 
estTR = zeros(params.nStates); estEM = zeros(params.nStates);

% Fit in n-folds
for iFold = 1:params.nFolds

    data = foldedData(:,iFold);

    % Let's run a k-means clustering to get initial UP and DOWN clusters
    [cluInds] = kmeans(data, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

    % Now we assign to each cluster the cluster centres, i.e. the mean firing rates
    tmp1 = data(cluInds==1); tmp2 = data(cluInds==2);
    %Assign states correctly
    if mean(tmp1) > mean(tmp2)
        initUp = data(cluInds==1); initDown = data(cluInds==2);
    else
        initUp = data(cluInds==2); initDown = data(cluInds==1);
    end


    % Generate the initial emission matrix using these normalised rates from above. Alternatively, you could draw them from a poisson distribution using the rates
    % as the respective lambdas. The emission matrix is an nStates x nSymbols matrix that holds the probability of a particular symbol being emitted by a particular
    % state. So naturally, during a down state the probability of emitting a spike should be low and vice versa.
    % You could also just randomly initialise this, but training is faster if these are empirically informed.
    l1 = mean(initUp); l2 = mean(initDown); lambda = [l1 l2]./(l1+l2);
    initEmis = [lambda(2) lambda(1); lambda(1) lambda(2)];

    % Generate the initial sequence guess. You could also generate this from the cluster indices but its more complicated. The sequence is the sequence of symbols
    % emitted by the 2 underlying hidden states.
    initSeqTraining = ones(1,length(data));
    initSeqTraining(data>=mean(data))=2;

    % Train the HMM
    [tmp1,tmp2] = hmmtrain(initSeqTraining,initTransMat,initEmis,'Verbose',true,'Tolerance',params.tol);
    estTR = estTR+tmp1; estEM = estEM+tmp2;

end

estTR = estTR./params.nFolds; estEM = estEM./params.nFolds; % These are the learned transition and emission probability matrices

%% Testing and validation

% Create the initial sequence guess from the testing data
initSeqTesting = ones(1,length(testingData));
initSeqTesting(testingData>=mean(testingData))=2;

% Compute the log-likelihood 
[~,lll] = hmmdecode(initSeqTesting, estTR, estEM);

% Compute the Akaike Information Criterion and Bayesian Information Criterion as goodness-of-fit measures for model evaluation (Well, this is the ground truth
% so....)
[~, AIC, BIC] = compute_goodness_of_fit(initSeqTesting, estTR, estEM, lll);

%% Extract statistics

% Now compute the posterior path probabilities and most likely state sequence for the entire chunk of data so that we can compute the distributions of the UP
% and DOWN states and verify that their means are what we specified in our simulation!
% I know that this contains the data we used to train. But here we're not looking for goodness of fit. Just to compute the durations. The goodness of fit above
% is PURELY COMPUTED ON THE UNSEEN TESTING DATA

% Create the initial sequence guess from the testing data
initSeq = ones(1,length(instFiringRate));
initSeq(instFiringRate>=mean(instFiringRate))=2;

% Compute the posterior path probabilities using the learned matrices
[alphaFull] = hmmdecode(initSeq, estTR, estEM);

% Compute the most likely path sequence using the Viterbi algorithm
[stateSeqFull] = hmmviterbi(initSeq, estTR, estEM);

% Correct the most likely path sequence using history
stateSeqFull = smoothOverChunks(stateSeqFull-1,params.history,params.binnedFs); %Uh so the functions smoothOverChunks() and getBinaryChunkDurations() are 
% adapted from code I wrote in Tübingen to detect various things like eyesOpen-eyesClosed, beta bursts, OKN vs Piecemeal etc. Therefore they only take in 
% 0s and 1s. That's why we subtract 1 from the output StateSequence

% Compute the durations of the UP/DOWN states
[dursDownState, dursUpState] = getBinaryChunkDurations(stateSeqFull);
dursDownState = dursDownState./params.binnedFs; dursUpState = dursUpState./params.binnedFs;

mUp = mean(dursUpState); 
mDown = mean(dursDownState);

% NB - Report this as rate +/- jitter!!! VERY IMPORTANT

% Compute the cycle frequencies. Cycle frequencies are defined as the inverse of the sum of the ith UP state and the (i+1)th DOWN state

nCycles = min(length(dursUpState), length(dursDownState));
cycleFreqs = 1./(dursUpState(1:nCycles) + dursDownState(1:nCycles));

% Fit a Gamma distribution
gamPars = gamfit(cycleFreqs);
freqBins = 0:0.1:max(cycleFreqs)+1;
altFreqPDF = gampdf(freqBins,gamPars(1),gamPars(2));


%% Plotting

% DON'T FUCK WITH THIS

binsForFittingTime = 0:params.history*2:max([dursDownState dursUpState]);

figure;
sgtitle('Simulated Data')
subplot(2,2,[1 2])
yyaxis left
for iSpk = 1:length(spikeTimes)
    st = spikeTimes{iSpk}(spikeTimes{iSpk}<=params.chunkSize);
    if ~isempty(st)
        plot([st; st], [iSpk iSpk+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spikeTimes)+1])
ylabel('Neuron ID')
yyaxis right
plot(bins, (alphaFull(1,:)),'LineWidth',1.5)
hold on
plot(bins,stateSeqFull,'-g','LineWidth',1.5)
ylabel('P(UP State)')
xlabel('time [s]')
title('Spiking Raster')
xlim([0 20])
box off;

subplot(2,2,3)
histogram(dursDownState,binsForFittingTime,'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpState,binsForFittingTime,'Normalization', 'pdf','FaceAlpha',0.25)
vline(mean(dursUpState),'--r'); vline(mean(dursDownState),'--b')
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
title('Duration distribution of hidden states')
box off;

subplot(2,2,4)
histogram(cycleFreqs,freqBins,'FaceColor',[0.75 0.75 0.75],'Normalization','pdf')
hold on
plot(freqBins,altFreqPDF,'LineWidth',2)
box off;
vline(mean(cycleFreqs),'--r')
xlabel('alternation frequency [Hz]')
ylabel('probability density []')
title('UP/DOWN Cycle Alternation')