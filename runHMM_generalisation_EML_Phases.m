clear all
clc
close all

dbstop if error

% Script to fit HMM to spiking data, compute durations, perform statistical tests, plot and save figs. To compare Wakefulness vs Light Anaesthesia
% Pre-stimulation and Light Anaesthesia post FAHA stimulation
%
% AD. NS. 2023

%% Ask user for input for the session #

userInput = inputdlg('Enter Session number:', 'Integer Input', [1 40]);
session = str2double(userInput);

if ~isempty(session) && isnumeric(session) && session >= 1 && session <= 4
    if session == 3
        userInput = inputdlg('Enter FAHA number (2,3,4):', 'Integer Input', [1 40]); % In session 3 we performed 3 blocks of FAHA Stimulation
        postFAHA = str2double(userInput);
    else
        postFAHA = 2;
    end

    h=msgbox(['Analysing Session: ' num2str(session)], 'Valid Input');
    pause(3);
    delete(h);
else
    h=msgbox('Session not parsed yet.', 'Invalid Input');
    pause(3);
    delete(h);
end

%% load dataset and set params

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' num2str(session) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' num2str(session) '\21'])
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' num2str(session) '\' '2' num2str(postFAHA)])
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.025; %s. Bin width to generate PSTH. Bin spikes in 25ms bins. Could also use an alpha kernel to eliminate "history" efffects.
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't cry. This was when we had 2 arrays, 96x2.
params.firingThresh = 0.05; %Hz. Atleast fire some spikes.
params.binnedFs = 1/params.binWidth;
params.chunkSize = 20; %s. To plot representative rasters
params.channelThreshold = 15; % To remove cross-channel movement artifacts.
params.offset = 2; % Machine lag can occur.
params.nStates = 2; % For the HMM
params.nIter = 1e3; % max iterations for convergence
params.tol = 1e-6; % convergence threshold.
params.history = 0.05; %Seconds. Various authors report using between 50-100ms history window to smooth over very brief bursts of noise.
params.nFolds = 10; % For training the HMM on LS PreStim RS
params.holdOut = 0.2;% percent
params.nPhases = 3;

%% If session = 4, clean the dataset again. There were serious motion artifacts in the awake block. No such artifacts during anaesthesia

if session == 4

    rasterDataAwake = removeCommonArtifacts(params,rasterDataAwake);

end

%% Do shit

% Remove chutiya neurons

% FIXED POPULATION ANALYSIS. Select good firing neurons based on a firing rate threshold from the awake block. Use those neurons also in the anaesthetised
% blocks. This way we can track the change in population activity.

firingRatesAwake = sum(rasterDataAwake,2)/tAwake(end);
idx = firingRatesAwake >= params.firingThresh;
nValChans = sum(idx);

valRasterAwake = rasterDataAwake(idx,:);
valRasterLSPre = rasterDataLSPre(idx,:);
valRasterLSPost = rasterDataLSPost(idx,:);

% Collect spike counts by binning

binsAwake = 0:params.binWidth:tAwake(end);
binsLSPre = 0:params.binWidth:tLSPre(end);
binsLSPost = 0:params.binWidth:tLSPost(end);

spikeCountsAwake = zeros(nValChans,length(binsAwake));
spikeCountsLSPre = zeros(nValChans,length(binsLSPre));
spikeCountsLSPost = zeros(nValChans,length(binsLSPost));

spkTimesAwake = cell(1,nValChans);
spkTimesLSPre = cell(1,nValChans);
spkTimesLSPost = cell(1,nValChans);

for iChan = 1:nValChans

    idx = valRasterAwake(iChan,:)==1;
    spkTimesAwake{iChan} = tAwake(idx);
    spikeCountsAwake(iChan,:) = histc(spkTimesAwake{iChan},binsAwake)./params.binWidth;
    clear idx;

    idx = valRasterLSPre(iChan,:)==1;
    spkTimesLSPre{iChan} = tLSPre(idx);
    spikeCountsLSPre(iChan,:) = histc(spkTimesLSPre{iChan},binsLSPre)./params.binWidth;
    clear idx;

    idx = valRasterLSPost(iChan,:)==1;
    spkTimesLSPost{iChan} = tLSPost(idx);
    spikeCountsLSPost(iChan,:) = histc(spkTimesLSPost{iChan},binsLSPost)./params.binWidth;
    clear idx;

end

% Sort neurons by firing rate

spkRatesAwake = nanmean(spikeCountsAwake,2);
spkRatesLSPre = nanmean(spikeCountsLSPre,2);
spkRatesLSPost = nanmean(spikeCountsLSPost,2);

[~,idx1] = sort(spkRatesAwake,'ascend');
[~,idx2] = sort(spkRatesLSPre,'ascend');
[~,idx3] = sort(spkRatesLSPost,'ascend');

sortedSpkTimesAwake = cell(size(spkTimesAwake));
sortedSpkTimesLSPre = cell(size(spkTimesLSPre));
sortedSpkTimesLSPost = cell(size(spkTimesLSPost));

for iSorted = 1:nValChans

    sortedSpkTimesAwake{iSorted} = spkTimesAwake{idx1(iSorted)};
    sortedSpkTimesLSPre{iSorted} = spkTimesLSPre{idx2(iSorted)};
    sortedSpkTimesLSPost{iSorted} = spkTimesLSPost{idx3(iSorted)};

end

spkTimesAwake = sortedSpkTimesAwake;
spkTimesLSPre = sortedSpkTimesLSPre;
spkTimesLSPost = sortedSpkTimesLSPost;

clear sortedSpkTimesAwake; clear sortedSpkTimesLSPre; clear sortedSpkTimesLSPost;

% Create the PSTH and normalise it between 0 and 1. Then constrain it between 1e-3 and 0.999. Avoid overflow shit.

frAwake = normalise(nanmean(spikeCountsAwake,1));
frAwake(frAwake==0) = 1e-3; frAwake(frAwake==1) = 0.999;
frLSPre = normalise(nanmean(spikeCountsLSPre,1));
frLSPre(frLSPre==0) = 1e-3; frLSPre(frLSPre==1) = 0.999;
frLSPost = normalise(nanmean(spikeCountsLSPost,1));
frLSPost(frLSPost==0) = 1e-3; frLSPre(frLSPost==1) = 0.999;

%% Split the data into Early, middle late phases

remAwake = mod(length(frAwake),params.nPhases);
frAwake = frAwake(1:end-remAwake);
chunkSize = length(frAwake)/params.nPhases;
frAwake = reshape(frAwake, chunkSize, params.nPhases);

remLSPre = mod(length(frLSPre),params.nPhases);
frLSPre = frLSPre(1:end-remLSPre);
chunkSize = length(frLSPre)/params.nPhases;
frLSPre = reshape(frLSPre, chunkSize, params.nPhases);

remLSPost = mod(length(frLSPost),params.nPhases);
frLSPost = frLSPost(1:end-remLSPost);
chunkSize = length(frLSPost)/params.nPhases;
frLSPost = reshape(frLSPost, chunkSize, params.nPhases);

%% Do the fitting on the LS Pre

% Generate a "random" initial transition matrix
initTransMat = 0.3+(0.7-0.3)*rand(params.nStates);

for iPhase = 1:params.nPhases

    % Create the initial guess sequences
    initSeqAwake = ones(1,length(frAwake(:,iPhase)));
    initSeqAwake(frAwake(:,iPhase)>=mean(frAwake(:,iPhase)))=2;

    initSeqLSPost = ones(1,length(frLSPost(:,iPhase)));
    initSeqLSPost(frLSPost(:,iPhase)>=mean(frLSPost(:,iPhase)))=2;

    rs=rng(1234); % Random seed for reproducibility

    % Partition LS Pre data for training and testing.

    % WE ASSUME LS PRE STIM DATA TO BE THE "GROUND TRUTH". This is because anaesthesia was turned on and allowed to stabilise before marking the start of the
    % resting state epoch. Under anaesthesia, the cortex reorganises spiking into periods of bursting (UP) and quiescence (DOWN). Our HMM model assumes that these 2
    % states emit 2 symbols (2 = firing, 1 = no firing). Because there were no perturbations performed, we take this epoch to be our "ground truth" for UP/DOWN
    % state patterns.

    % Partition the data

    nTrainingSamps = floor((1-params.holdOut)*length(frLSPre(:,iPhase)));
    trainingData = frLSPre(1:nTrainingSamps,iPhase);
    testingData = frLSPre(nTrainingSamps+1:end,iPhase);

    %Fit in nFolds and take the average estimate

    foldSize = floor(length(trainingData)/params.nFolds);
    remainder = mod(length(trainingData), params.nFolds);
    trainingData = trainingData(1:end-remainder);

    foldedData = reshape(trainingData, length(trainingData)/params.nFolds, params.nFolds);
    estTRLSPreChunk = zeros(params.nStates); estEMLSPreChunk = zeros(params.nStates);

    for iFold = 1:params.nFolds

        data = foldedData(:,iFold);
        % Run 2 cluster k-means to extract the initial clusters
        [cluIndsLSPre] = kmeans(data, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

        tmp1 = frLSPre(cluIndsLSPre==1); tmp2 = frLSPre(cluIndsLSPre==2);
        %Assig{iPhase}n states correctly
        if mean(tmp1) > mean(tmp2)
            initUpLSPre = frLSPre(cluIndsLSPre==1); initDownLSPre = frLSPre(cluIndsLSPre==2);
        else
            initUpLSPre = frLSPre(cluIndsLSPre==2); initDownLSPre = frLSPre(cluIndsLSPre==1);
        end
        % Generate the initial emission matrix using normalised rates as the poisson lambdas
        l1LSPre = mean(initUpLSPre); l2LSPre = mean(initDownLSPre); lambdaLSPre = [l1LSPre l2LSPre]./(l1LSPre+l2LSPre);
        initEmisLSPre = [lambdaLSPre(2) lambdaLSPre(1); lambdaLSPre(1) lambdaLSPre(2)];
        initSeqLSPreTraining = ones(1,length(data));
        initSeqLSPreTraining(data>=mean(data))=2;
        % Train the HMM
        [tmp1,tmp2] = hmmtrain(initSeqLSPreTraining,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol);
        estTRLSPreChunk = estTRLSPreChunk+tmp1; estEMLSPreChunk = estEMLSPreChunk+tmp2;

    end

    estTRLSPre{iPhase} = estTRLSPreChunk./params.nFolds; estEMLSPre{iPhase} = estEMLSPreChunk./params.nFolds;

    % Do validation and generalisation

    % Because we assume that RS PreStim LS is our "ground truth", we use the fitted TR and EM matrices from the training to generalise to Wakefulness and
    % post-stimulation anaesthesia epochs. We want to see how well our 2-state HMM can fit the data before and after perturbation.
    % NB - Cortical spiking activity is desynchronised during wakefulness, i.e. each neuron does its own stuff. Under anaesthesia, they become coherent. Hypothesis
    % and initial observations show that post FAHA stimulation, the dynamics of the UP/DOWN states may change and tend towards more desynchrony akin to awake state.
    % This is what we test here.

    initSeqLSPreTesting = ones(1,length(testingData));
    initSeqLSPreTesting(testingData>=mean(testingData))=2;

    % Get the path probability sequences and most likely state sequence
    [alphaAwake{iPhase},lllAwake{iPhase}] = hmmdecode(initSeqAwake, estTRLSPre{iPhase}, estEMLSPre{iPhase});
    [~, AICAwake{iPhase}, BICAwake{iPhase}] = compute_goodness_of_fit(initSeqAwake, estTRLSPre{iPhase}, estEMLSPre{iPhase}, lllAwake{iPhase});
    [stateSeqAwake{iPhase}] = hmmviterbi(initSeqAwake, estTRLSPre{iPhase}, estEMLSPre{iPhase});

    [alphaLSPreTesting{iPhase},lllLSPre{iPhase}] = hmmdecode(initSeqLSPreTesting, estTRLSPre{iPhase}, estEMLSPre{iPhase});
    [~, AICLSPre{iPhase}, BICLSPre{iPhase}] = compute_goodness_of_fit(initSeqLSPreTesting, estTRLSPre{iPhase}, estEMLSPre{iPhase}, lllLSPre{iPhase});
    [stateSeqLSPreTesting{iPhase}] = hmmviterbi(initSeqLSPreTesting, estTRLSPre{iPhase}, estEMLSPre{iPhase});

    [alphaLSPost{iPhase},lllLSPost{iPhase}] = hmmdecode(initSeqLSPost, estTRLSPre{iPhase}, estEMLSPre{iPhase});
    [~, AICLSPost{iPhase}, BICLSPost{iPhase}] = compute_goodness_of_fit(initSeqLSPost, estTRLSPre{iPhase}, estEMLSPre{iPhase}, lllLSPost{iPhase});
    [stateSeqLSPost{iPhase}] = hmmviterbi(initSeqLSPost, estTRLSPre{iPhase}, estEMLSPre{iPhase});

    % Get the full state sequence for LSPre
    initSeqLSPre = ones(1,length(frLSPre(:,iPhase)));
    initSeqLSPre(frLSPre(:,iPhase)>=mean(frLSPre(:,iPhase)))=2;
    [alphaLSPre{iPhase}] = hmmdecode(initSeqLSPre, estTRLSPre{iPhase}, estEMLSPre{iPhase});
    [stateSeqLSPre{iPhase}] = hmmviterbi(initSeqLSPre, estTRLSPre{iPhase}, estEMLSPre{iPhase});

    % Fix very small chunks using the "history" parameter
    stateSeqAwake{iPhase} = smoothOverChunks(stateSeqAwake{iPhase}-1,params.history,params.binnedFs);
    stateSeqLSPre{iPhase} = smoothOverChunks(stateSeqLSPre{iPhase}-1,params.history,params.binnedFs);
    stateSeqLSPost{iPhase} = smoothOverChunks(stateSeqLSPost{iPhase}-1,params.history,params.binnedFs);

    % Get state durations
    [dursDownStateAwake{iPhase}, dursUpStateAwake{iPhase}] = getBinaryChunkDurations(stateSeqAwake{iPhase});
    dursDownStateAwake{iPhase} = dursDownStateAwake{iPhase}./params.binnedFs; dursUpStateAwake{iPhase} = dursUpStateAwake{iPhase}./params.binnedFs;

    [dursDownStateLSPre{iPhase}, dursUpStateLSPre{iPhase}] = getBinaryChunkDurations(stateSeqLSPre{iPhase});
    dursDownStateLSPre{iPhase} = dursDownStateLSPre{iPhase}./params.binnedFs; dursUpStateLSPre{iPhase} = dursUpStateLSPre{iPhase}./params.binnedFs;

    [dursDownStateLSPost{iPhase}, dursUpStateLSPost{iPhase}] = getBinaryChunkDurations(stateSeqLSPost{iPhase});
    dursDownStateLSPost{iPhase} = dursDownStateLSPost{iPhase}./params.binnedFs; dursUpStateLSPost{iPhase} = dursUpStateLSPost{iPhase}./params.binnedFs;

    % Compute ACFs and get synchrony index

    [acAwake{iPhase},~,SIAwake{iPhase}] = sig_autocorr(frAwake(:,iPhase),params.binnedFs,params.history*30);
    [acLSPre{iPhase},~,SILSPre{iPhase}] = sig_autocorr(frLSPre(:,iPhase),params.binnedFs,params.history*30);
    [acLSPost{iPhase},lags,SILSPost{iPhase}] = sig_autocorr(frLSPost(:,iPhase),params.binnedFs,params.history*30);

end

%% Plot and shit

for iPhase = 1:params.nPhases

    if iPhase == 1

        phase = 'Early';

    elseif iPhase == 2

        phase = 'Middle';

    elseif iPhase == 3

        phase = 'Late';

    end

    figure('units','normalized','outerposition',[0 0 1 1]);
    sgtitle(['Measures of synchrony - Session # - ' num2str(session) '. Phase - ' phase])

    subplot(1,2,1)
    title('Autocorrelation Functions')
    plot(lags,acAwake{iPhase},'LineWidth',1.5)
    hold on
    plot(lags,acLSPre{iPhase},'LineWidth',1.5)
    plot(lags,acLSPost{iPhase},'LineWidth',1.5)
    box off;
    xlabel('lags [s]')
    ylabel('z-scored ACF []')
    legend('QW','RS PreStim LS','RS PostStim LS')

    subplot(1,2,2)
    title('Measure of Synchrony')
    plot([1 2 3],1./[SIAwake{iPhase} SILSPre{iPhase} SILSPost{iPhase}],'-k','LineWidth',1.5,'Marker','o','MarkerSize',10)
    xticks([1 2 3])
    % Set the x-axis tick labels
    xticklabels({'QW', 'RS PreStim LS', 'RS PostStim LS'})
    box off;
    ylabel('Synchrony Index []')
    xlim([0.5 3.5])

    %% Do some statistics

    % Do KDE for the durations and perform ranksum test

    binSize = params.history/2.5;
    binsForFittingTime = 0:binSize:2;

    tic;
    [upDursLSPre,~] = ksdensity(dursUpStateLSPre{iPhase},binsForFittingTime,'Support','positive','BoundaryCorrection','log'); [downDursLSPre,~] = ksdensity(dursDownStateLSPre{iPhase},binsForFittingTime,'Support','positive','BoundaryCorrection','log','Function','pdf');
    [upDursLSPost,~] = ksdensity(dursUpStateLSPost{iPhase},binsForFittingTime,'Support','positive','BoundaryCorrection','log'); [downDursLSPost,~] = ksdensity(dursDownStateLSPost{iPhase},binsForFittingTime,'Support','positive','BoundaryCorrection','log','Function','pdf');
    toc;

    [pVal{iPhase}(1),sig{iPhase}(1)] = ranksum(dursUpStateLSPre{iPhase},dursUpStateLSPost{iPhase});
    [pVal{iPhase}(2),sig{iPhase}(2)] = ranksum(dursDownStateLSPre{iPhase},dursDownStateLSPost{iPhase});

    % Get cycle alternation frequency and perform KDE and ranksum test

    nCyclesLSPre = min(length(dursUpStateLSPre{iPhase}), length(dursDownStateLSPre{iPhase}));
    cycleDursLSPre = 1./(dursUpStateLSPre{iPhase}(1:nCyclesLSPre) + dursDownStateLSPre{iPhase}(1:nCyclesLSPre));

    nCyclesLSPost = min(length(dursUpStateLSPost{iPhase}), length(dursDownStateLSPost{iPhase}));
    cycleDursLSPost = 1./(dursUpStateLSPost{iPhase}(1:nCyclesLSPost) + dursDownStateLSPost{iPhase}(1:nCyclesLSPost));

    binSize = 1/(100*params.history);
    binsForFittingFreqs = 0:binSize:(0.5/params.history); % Max frequency corresponds to smallest possible state duration

    tic;
    cycleFreqsLSPre = ksdensity(cycleDursLSPre,binsForFittingFreqs,'Support','positive','BoundaryCorrection','log','Function','pdf');
    cycleFreqsLSPost = ksdensity(cycleDursLSPost,binsForFittingFreqs,'Support','positive','BoundaryCorrection','log','Function','pdf');
    toc;

    [pVal{iPhase}(3),sig{iPhase}(3)] = ranksum(cycleDursLSPre,cycleDursLSPost);

    figure('units','normalized','outerposition',[0 0 1 1]);
    title(['Phase - ' phase])
    bar([AICAwake{iPhase} AICLSPre{iPhase} AICLSPost{iPhase}; NaN NaN NaN])
    ax = gca;
    ax.XLim = [0.5 1.5];
    box off
    ylabel('model (in)fidelity - AIC []')
    legend('Quiet Wakefulness','Light Anaesthesia - PreStim', 'Light Anaesthesia - PostStim')

    figure('units','normalized','outerposition',[0 0 1 1]);
    plot(binsForFittingFreqs(2:end-1),cycleFreqsLSPre(2:end-1),'LineWidth',2)
    hold on
    plot(binsForFittingFreqs(2:end-1),cycleFreqsLSPost(2:end-1),'LineWidth',2)
    box off
    vline(mean(cycleDursLSPre),'--b');vline(mean(cycleDursLSPost),'--r');
    xlabel('UP/DOWN cycle frequency [Hz]')
    ylabel('normalised density []')
    legend('LS PreStim RS','LS PostStim RS')
    title(['Cycle Frequency Change - Session # - ' num2str(session) '. Phase - ' phase])

    figure('units','normalized','outerposition',[0 0 1 1]);
    sgtitle(['Session # : ' num2str(session) '. Phase - ' phase])
    subplot(1,2,1)
    plot(binsForFittingTime(2:end-1),upDursLSPre(2:end-1),'LineWidth',2)
    hold on
    plot(binsForFittingTime(2:end-1),upDursLSPost(2:end-1),'LineWidth',2)
    xlabel('state duration [s]')
    ylabel('normalised density []')
    title('UP States'); box off;
    legend('LS PreStim RS','LS PostStim RS')

    subplot(1,2,2)
    plot(binsForFittingTime(2:end-1),downDursLSPre(2:end-1),'LineWidth',2)
    hold on
    plot(binsForFittingTime(2:end-1),downDursLSPost(2:end-1),'LineWidth',2)
    xlabel('state duration [s]')
    ylabel('normalised density []')
    title('DOWN States'); box off;
    legend('LS PreStim RS','LS PostStim RS')

end

%% Plot the shit

figure('units','normalized','outerposition',[0 0 1 1]);
sgtitle('Comparison of duration distributions of the UP/DOWN States')

subplot(3,2,1)
histogram(dursUpStateAwake{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursUpStateLSPre{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursUpStateLSPost{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('UP State - Phase - Early')
box off;

subplot(3,2,2)
histogram(dursDownStateAwake{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursDownStateLSPre{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursDownStateLSPost{1},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('Down State - Phase - Early')
box off;

subplot(3,2,3)
histogram(dursUpStateAwake{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursUpStateLSPre{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursUpStateLSPost{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('UP State - Phase - Middle')
box off;

subplot(3,2,4)
histogram(dursDownStateAwake{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursDownStateLSPre{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursDownStateLSPost{2},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('Down State - Phase - Middle')
box off;

subplot(3,2,5)
histogram(dursUpStateAwake{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursUpStateLSPre{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursUpStateLSPost{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('UP State - Phase - Late')
box off;

subplot(3,2,6)
histogram(dursDownStateAwake{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.75)
hold on
histogram(dursDownStateLSPre{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.5)
histogram(dursDownStateLSPost{3},binsForFittingTime(1:3:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
legend('QW','LS PreStim RS', 'LS PostSim RS')
title('Down State - Phase - Late')
box off;

state(1).durations(1).UP = dursUpStateAwake{1}; state(1).durations(1).DOWN = dursDownStateAwake{1};
state(1).durations(2).UP = dursUpStateAwake{2}; state(1).durations(2).DOWN = dursDownStateAwake{2};
state(1).durations(3).UP = dursUpStateAwake{3}; state(1).durations(3).DOWN = dursDownStateAwake{3};

state(2).durations(1).UP = dursUpStateLSPre{1}; state(2).durations(1).DOWN = dursDownStateLSPre{1};
state(2).durations(2).UP = dursUpStateLSPre{2}; state(2).durations(2).DOWN = dursDownStateLSPre{2};
state(2).durations(3).UP = dursUpStateLSPre{3}; state(2).durations(3).DOWN = dursDownStateLSPre{3};

state(3).durations(1).UP = dursUpStateLSPost{1}; state(3).durations(1).DOWN = dursDownStateLSPost{1};
state(3).durations(2).UP = dursUpStateLSPost{2}; state(3).durations(2).DOWN = dursDownStateLSPost{2};
state(3).durations(3).UP = dursUpStateLSPost{3}; state(3).durations(3).DOWN = dursDownStateLSPost{3};

%% Save all the figures

cd('C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\Results')

% Create a sub-folder to hold the genWithFolds results separately. DO NOT PUT THEM IN THE MAIN DIRECTORY!
mkdir genWithFolds
cd genWithFolds
mkdir Phases
cd Phases
mkdir ft0-05
cd ft0-05
mkdir(num2str(session))
cd(num2str(session))
if session == 3
    mkdir(num2str(postFAHA))
    cd(num2str(postFAHA))
end

% Get all open figure handles
figure_handles = findall(0, 'Type', 'figure');

% Loop through the figure handles and save each figure
for iFig = 1:length(figure_handles)
    fig = figure_handles(iFig);
    % Set the desired file name and format for the saved figure
    file_name1 = sprintf('Fig_%d.png', iFig);
    file_name2 = sprintf('Fig_%d.fig', iFig);

    % Save the figure
    saveas(fig, file_name1);
    saveas(fig, file_name2);
end
save('durations.mat','state','-v7.3')
save('synchronyIndex.mat','SIAwake','SILSPre','SILSPost','-v7.3')
save('statistics.mat','pVal','-v7.3')
pause(3)
close all
