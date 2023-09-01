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

cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\21'])
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\' '2' num2str(postFAHA)])
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.025; %s. Bin width to generate PSTH. Bin spikes in 25ms bins. Could also use an alpha kernel to eliminate "history" efffects.
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't cry. This was when we had 2 arrays, 96x2.
params.firingThresh = 0.125; %Hz. Atleast fire some spikes.
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

    idx = find(valRasterAwake(iChan,:)==1);
    spkTimesAwake{iChan} = tAwake(idx);
    spikeCountsAwake(iChan,:) = histc(spkTimesAwake{iChan},binsAwake)./params.binWidth;
    clear idx;

    idx = find(valRasterLSPre(iChan,:)==1);
    spkTimesLSPre{iChan} = tLSPre(idx);
    spikeCountsLSPre(iChan,:) = histc(spkTimesLSPre{iChan},binsLSPre)./params.binWidth;
    clear idx;

    idx = find(valRasterLSPost(iChan,:)==1);
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

%% Now fit the shit

% Generate a "random" initial transition matrix
initTransMat = 0.3+(0.7-0.3)*rand(params.nStates);

% Create the initial guess sequences
initSeqAwake = ones(1,length(frAwake));
initSeqAwake(frAwake>=mean(frAwake))=2;

initSeqLSPost = ones(1,length(frLSPost));
initSeqLSPost(frLSPost>=mean(frLSPost))=2;

rng(1234); % Random seed for reproducibility

% Partition LS Pre data for training and testing. 

% WE ASSUME LS PRE STIM DATA TO BE THE "GROUND TRUTH". This is because anaesthesia was turned on and allowed to stabilise before marking the start of the
% resting state epoch. Under anaesthesia, the cortex reorganises spiking into periods of bursting (UP) and quiescence (DOWN). Our HMM model assumes that these 2
% states emit 2 symbols (2 = firing, 1 = no firing). Because there were no perturbations performed, we take this epoch to be our "ground truth" for UP/DOWN
% state patterns.

% Partition the data

nTrainingSamps = floor((1-params.holdOut)*length(frLSPre));
trainingData = frLSPre(1:nTrainingSamps);
testingData = frLSPre(nTrainingSamps+1:end);

%Fit in nFolds and take the average estimate

foldSize = floor(length(trainingData)/params.nFolds);
remainder = mod(length(trainingData), params.nFolds);
trainingData = trainingData(1:end-remainder);

foldedData = reshape(trainingData, length(trainingData)/params.nFolds, params.nFolds);
estTRLSPre = zeros(params.nStates); estEMLSPre = zeros(params.nStates);

for iFold = 1:params.nFolds

    data = foldedData(:,iFold);
    % Run 2 cluster k-means to extract the initial clusters
    [cluIndsLSPre] = kmeans(data, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

    tmp1 = frLSPre(cluIndsLSPre==1); tmp2 = frLSPre(cluIndsLSPre==2);
    %Assign states correctly
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
    estTRLSPre = estTRLSPre+tmp1; estEMLSPre = estEMLSPre+tmp2;

end

estTRLSPre = estTRLSPre./params.nFolds; estEMLSPre = estEMLSPre./params.nFolds;

%% Do validation and generalisation

% Because we assume that RS PreStim LS is our "ground truth", we use the fitted TR and EM matrices from the training to generalise to Wakefulness and
% post-stimulation anaesthesia epochs. We want to see how well our 2-state HMM can fit the data before and after perturbation.
% NB - Cortical spiking activity is desynchronised during wakefulness, i.e. each neuron does its own stuff. Under anaesthesia, they become coherent. Hypothesis
% and initial observations show that post FAHA stimulation, the dynamics of the UP/DOWN states may change and tend towards more desynchrony akin to awake state.
% This is what we test here. 

initSeqLSPreTesting = ones(1,length(testingData));
initSeqLSPreTesting(testingData>=mean(testingData))=2;

% Get the path probability sequences and most likely state sequence
[alphaAwake,lllAwake] = hmmdecode(initSeqAwake, estTRLSPre, estEMLSPre);
[~, AICAwake, BICAwake] = compute_goodness_of_fit(initSeqAwake, estTRLSPre, estEMLSPre, lllAwake);
[stateSeqAwake] = hmmviterbi(initSeqAwake, estTRLSPre, estEMLSPre);

[alphaLSPreTesting,lllLSPre] = hmmdecode(initSeqLSPreTesting, estTRLSPre, estEMLSPre);
[~, AICLSPre, BICLSPre] = compute_goodness_of_fit(initSeqLSPreTesting, estTRLSPre, estEMLSPre, lllLSPre);
[stateSeqLSPreTesting] = hmmviterbi(initSeqLSPreTesting, estTRLSPre, estEMLSPre);

[alphaLSPost,lllLSPost] = hmmdecode(initSeqLSPost, estTRLSPre, estEMLSPre);
[~, AICLSPost, BICLSPost] = compute_goodness_of_fit(initSeqLSPost, estTRLSPre, estEMLSPre, lllLSPost);
[stateSeqLSPost] = hmmviterbi(initSeqLSPost, estTRLSPre, estEMLSPre);

% Get the full state sequence for LSPre
initSeqLSPre = ones(1,length(frLSPre));
initSeqLSPre(frLSPre>=mean(frLSPre))=2;
[alphaLSPre] = hmmdecode(initSeqLSPre, estTRLSPre, estEMLSPre);
[stateSeqLSPre] = hmmviterbi(initSeqLSPre, estTRLSPre, estEMLSPre);

% Fix very small chunks using the "history" parameter
stateSeqAwake = smoothOverChunks(stateSeqAwake-1,params.history,params.binnedFs);
stateSeqLSPre = smoothOverChunks(stateSeqLSPre-1,params.history,params.binnedFs);
stateSeqLSPost = smoothOverChunks(stateSeqLSPost-1,params.history,params.binnedFs);

% Get state durations
[dursDownStateAwake, dursUpStateAwake] = getBinaryChunkDurations(stateSeqAwake);
dursDownStateAwake = dursDownStateAwake./params.binnedFs; dursUpStateAwake = dursUpStateAwake./params.binnedFs;

[dursDownStateLSPre, dursUpStateLSPre] = getBinaryChunkDurations(stateSeqLSPre);
dursDownStateLSPre = dursDownStateLSPre./params.binnedFs; dursUpStateLSPre = dursUpStateLSPre./params.binnedFs;

[dursDownStateLSPost, dursUpStateLSPost] = getBinaryChunkDurations(stateSeqLSPost);
dursDownStateLSPost = dursDownStateLSPost./params.binnedFs; dursUpStateLSPost = dursUpStateLSPost./params.binnedFs;

%% Plot ACFs and get synchrony index

[acAwake,~,SIAwake] = sig_autocorr(frAwake,params.binnedFs,params.history*30);
[acLSPre,~,SILSPre] = sig_autocorr(frLSPre,params.binnedFs,params.history*30);
[acLSPost,lags,SILSPost] = sig_autocorr(frLSPost,params.binnedFs,params.history*30);

figure;
sgtitle(['Measures of synchrony - Session # - ' num2str(session)])

subplot(1,2,1)
title('Autocorrelation Functions')
plot(lags,acAwake,'LineWidth',1.5)
hold on
plot(lags,acLSPre,'LineWidth',1.5)
plot(lags,acLSPost,'LineWidth',1.5)
box off;
xlabel('lags [s]')
ylabel('z-scored ACF []')
legend('QW','RS PreStim LS','RS PostStim LS')

subplot(1,2,2)
title('Measure of Synchrony')
plot([1 2 3],1./[SIAwake SILSPre SILSPost],'-k','LineWidth',1.5,'Marker','o','MarkerSize',10)
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
[upDursLSPre,~] = ksdensity(dursUpStateLSPre,binsForFittingTime,'Support','positive','BoundaryCorrection','log'); [downDursLSPre,~] = ksdensity(dursDownStateLSPre,binsForFittingTime,'Support','positive','BoundaryCorrection','log','Function','pdf');
[upDursLSPost,~] = ksdensity(dursUpStateLSPost,binsForFittingTime,'Support','positive','BoundaryCorrection','log'); [downDursLSPost,~] = ksdensity(dursDownStateLSPost,binsForFittingTime,'Support','positive','BoundaryCorrection','log','Function','pdf');
toc;

[pVal(1),sig(1)] = ranksum(dursUpStateLSPre,dursUpStateLSPost);
[pVal(2),sig(2)] = ranksum(dursDownStateLSPre,dursDownStateLSPost);

% Get cycle alternation frequency and perform KDE and ranksum test

nCyclesLSPre = min(length(dursUpStateLSPre), length(dursDownStateLSPre));
cycleDursLSPre = 1./(dursUpStateLSPre(1:nCyclesLSPre) + dursDownStateLSPre(1:nCyclesLSPre));

nCyclesLSPost = min(length(dursUpStateLSPost), length(dursDownStateLSPost));
cycleDursLSPost = 1./(dursUpStateLSPost(1:nCyclesLSPost) + dursDownStateLSPost(1:nCyclesLSPost));

binSize = 1/(100*params.history);
binsForFittingFreqs = 0:binSize:(0.5/params.history); % Max frequency corresponds to smallest possible state duration

tic;
cycleFreqsLSPre = ksdensity(cycleDursLSPre,binsForFittingFreqs,'Support','positive','BoundaryCorrection','log','Function','pdf');
cycleFreqsLSPost = ksdensity(cycleDursLSPost,binsForFittingFreqs,'Support','positive','BoundaryCorrection','log','Function','pdf');
toc;

[pVal(3),sig(3)] = ranksum(cycleDursLSPre,cycleDursLSPost);

figure;
bar([AICAwake AICLSPre AICLSPost; NaN NaN NaN])
ax = gca;
ax.XLim = [0.5 1.5];
box off
ylabel('model (in)fidelity - AIC []')
legend('Quiet Wakefulness','Light Anaesthesia - PreStim', 'Light Anaesthesia - PostStim')

figure;
plot(binsForFittingFreqs(2:end-1),cycleFreqsLSPre(2:end-1),'LineWidth',2)
hold on
plot(binsForFittingFreqs(2:end-1),cycleFreqsLSPost(2:end-1),'LineWidth',2)
box off
vline(mean(cycleDursLSPre),'--b');vline(mean(cycleDursLSPost),'--r');
xlabel('UP/DOWN cycle frequency [Hz]')
ylabel('normalised density []')
legend('LS PreStim RS','LS PostStim RS')
title(['Cycle Frequency Change - Session # - ' num2str(session)])

figure;
sgtitle(['Session # : ' num2str(session)])
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

%% Plot the shit

figure;
sgtitle('Quiet Wakefulness')
subplot(2,1,1)
yyaxis left
for iSpk = 1:length(spkTimesAwake)
    st = spkTimesAwake{iSpk}(spkTimesAwake{iSpk}<params.chunkSize);
    if ~isempty(st)
        plot([st; st], [iSpk iSpk+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesAwake)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsAwake, (alphaAwake(1,:)),'LineWidth',1.5)
hold on
plot(binsAwake,stateSeqAwake,'-g','LineWidth',1.5)
ylabel('P(UP State)')
xlabel('time [s]')
title('Pre-stimulation Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateAwake,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateAwake,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
title('Duration distribution of hidden states')
box off;


figure;
sgtitle('Light Anaesthesia - Pre Stimulation')
subplot(2,1,1)
yyaxis left
for iSpk = 1:length(spkTimesLSPre)
    st = spkTimesLSPre{iSpk}(spkTimesLSPre{iSpk}<params.chunkSize);
    if ~isempty(st)
        plot([st; st], [iSpk iSpk+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesLSPre)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsLSPre, (alphaLSPre(1,:)),'LineWidth',1.5)
hold on
plot(binsLSPre,stateSeqLSPre,'-g','LineWidth',1.5)
ylabel('P(UP State)')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateLSPre,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateLSPre,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
title('Duration distribution of hidden states')
box off;


figure;
sgtitle('Light Anaesthesia - Post Stimulation')
subplot(2,1,1)
yyaxis left
for iSpk = 1:length(spkTimesLSPost)
    st = spkTimesLSPost{iSpk}(spkTimesLSPost{iSpk}<params.chunkSize);
    if ~isempty(st)
        plot([st; st], [iSpk iSpk+0.5],'-k')
    end
    hold on
end
xlim([0, 20])
ylim([0, length(spkTimesLSPost)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsLSPost, (alphaLSPost(1,:)),'LineWidth',1.5)
hold on
plot(binsLSPost,stateSeqLSPost,'-g','LineWidth',1.5)
ylabel('P(UP State)')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateLSPost,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateLSPost,binsForFittingTime(1:2:end), 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('probability density []')
title('Duration distribution of hidden states')
box off;

%% Save all the figures

cd('C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\Results\PFC')

% Create a sub-folder to hold the genWithFolds results separately. DO NOT PUT THEM IN THE MAIN DIRECTORY!
mkdir genWithFolds
cd genWithFolds
mkdir ft0-125
cd ft0-125
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
    file_name = sprintf('Fig_%d.png', iFig);
    
    % Save the figure
    saveas(fig, file_name);
end
save('durationFrequencyStatistcs.mat','pVal','sig','-v7.3')
pause(3)
close all
