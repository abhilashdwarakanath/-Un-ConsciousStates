clear all
clc
close all

dbstop if error

%% load dataset and set params

prompt = {'Enter session number (1 to 4)'};
dlgtitle = 'Session # - ';
dims = [1 40];
definput = {' '};
dataset = cell2mat(inputdlg(prompt,dlgtitle,dims,definput));

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' (dataset) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' (dataset) '\21'])
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' (dataset) '\22'])
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.025; %s
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't whine.
if str2double(dataset)==1
    params.firingThresh = 0.25; %Hz
elseif str2double(dataset)==4
    params.firingThresh = 7.5; %Hz
else
    params.firingThresh = 5;
end

params.binnedFs = 1/params.binWidth;
params.chunkSize = 100; %s
params.channelThreshold = 15;
params.offset = 2;
params.nStates = 2;
params.nIter = 1e3;
params.tol = 1e-6;
params.history = 0.075; %Seconds


%% If session = 4, clean the dataset again

if str2double(dataset) == 4
    rasterDataAwake = removeCommonArtifacts(params,rasterDataAwake);
end
%% Do shit

% Remove chutiya neurons

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

spkRatesAwake = nanmean(spikeCountsAwake,2);
spkRatesLSPre = nanmean(spikeCountsLSPre,2);
spkRatesLSPost = nanmean(spikeCountsLSPost,2);

[~,idx1] = sort(spkRatesAwake,'ascend');
[~,idx2] = sort(spkRatesLSPre,'ascend');
[~,idx3] = sort(spkRatesLSPost,'ascend');

sortedSpkTimesAwake = cell(size(spkTimesAwake));
sortedSpkTimesLSPre = cell(size(spkTimesLSPre));
sortedSpkTimesLSPost = cell(size(spkTimesLSPost));

% Loop through the index and copy the cells to the new cell array

for iSorted = 1:nValChans

    sortedSpkTimesAwake{iSorted} = spkTimesAwake{idx1(iSorted)};
    sortedSpkTimesLSPre{iSorted} = spkTimesLSPre{idx2(iSorted)};
    sortedSpkTimesLSPost{iSorted} = spkTimesLSPost{idx3(iSorted)};

end

spkTimesAwake = sortedSpkTimesAwake;
spkTimesLSPre = sortedSpkTimesLSPre;
spkTimesLSPost = sortedSpkTimesLSPost;

clear sortedSpkTimesAwake; clear sortedSpkTimesLSPre; clear sortedSpkTimesLSPost;

frAwake = normalise(nanmean(spikeCountsAwake,1));
frLSPre = normalise(nanmean(spikeCountsLSPre,1));
frLSPost = normalise(nanmean(spikeCountsLSPost,1));

%% Chunk them into 10s chunks

chunkSamples = params.chunkSize * params.binnedFs;

% Awake
nChunks = floor(length(frAwake)/chunkSamples);
newLength = nChunks * chunkSamples;
truncAwake = frAwake(1:newLength);
dataAwake = reshape(truncAwake, [chunkSamples, nChunks]);

% LSPre
nChunks = floor(length(frLSPre)/chunkSamples);
newLength = nChunks * chunkSamples;
truncLSPre = frLSPre(1:newLength);
dataLSPre = reshape(truncLSPre, [chunkSamples, nChunks]);

% LSPost
nChunks = floor(length(frLSPost)/chunkSamples);
newLength = nChunks * chunkSamples;
truncLSPost = frLSPost(1:newLength);
dataLSPost = reshape(truncLSPost, [chunkSamples, nChunks]);

%% Now fit the shit in MATLAB
tic;
% Set RND seed
rng(1234);

% Awake

% Awake

for iChunk = 1:size(dataAwake,2)

    % Get the segment out
    dataSegment = dataAwake(:,iChunk);
    dataSegment(dataSegment==0) = 0.0001;

    % Run k-means to get initial clusters
    [cluIndsAwake] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','iter');
    
     tmp1 = dataSegment(cluIndsAwake==1); tmp2 = dataSegment(cluIndsAwake==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpAwake = dataSegment(cluIndsAwake==1); initDownAwake = dataSegment(cluIndsAwake==2);

    else

        initUpAwake = dataSegment(cluIndsAwake==2); initDownAwake = dataSegment(cluIndsAwake==1);

    end


    % Turn lambda into lambda probs
    l1Awake = mean(initUpAwake); l2Awake = mean(initDownAwake); lambdaAwake = [l1Awake l2Awake]./(l1Awake+l2Awake);

    % Initialise TR matrix
    initTransMat = rand(params.nStates);

    % Initialise EM matrix
    initEmisAwake = [lambdaAwake(2) lambdaAwake(1); lambdaAwake(1) lambdaAwake(2)];

    % Initial sequence guess
    initSeqAwake = ones(1,length(dataSegment));
    initSeqAwake(dataSegment>=nanmean(dataSegment))=2;

    % Train the HMM using the Baum-Welch algorithm
    [estTRAwake{iChunk},estEMAwake{iChunk}] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaAwake(:,:,iChunk),lllAwake(iChunk)] = hmmdecode(initSeqAwake, estTRAwake{iChunk}, estEMAwake{iChunk});

    % Get goodness of fit using AIC and BIC
    [~, AICAwake(iChunk), BICAwake(iChunk)] = compute_goodness_of_fit(initSeqAwake, estTRAwake{iChunk}, estEMAwake{iChunk}, lllAwake(iChunk));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqAwake(:,iChunk)] = hmmviterbi(initSeqAwake, estTRAwake{iChunk}, estEMAwake{iChunk});

    % Fix small bursts using a history of 6 bins
    stateSeqAwake(:,iChunk) = smoothOverChunks(stateSeqAwake(:,iChunk)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateAwake{iChunk}, dursUpStateAwake{iChunk}] = getBinaryChunkDurations(stateSeqAwake(:,iChunk));
    dursDownStateAwake{iChunk} = dursDownStateAwake{iChunk}./params.binnedFs; dursUpStateAwake{iChunk} = dursUpStateAwake{iChunk}./params.binnedFs;

    clear initSeqAwake; clear initUpAwake; clear initDownAwake; clear initEmisAwake;

end

% LSPre

for iChunk = 1:size(dataLSPre,2)

    % Get the segment out
    dataSegment = dataLSPre(:,iChunk);
    dataSegment(dataSegment==0) = 0.0001;

    % Run k-means to get initial clusters
    [cluIndsLSPre] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','iter');
    
     tmp1 = dataSegment(cluIndsLSPre==1); tmp2 = dataSegment(cluIndsLSPre==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPre = dataSegment(cluIndsLSPre==1); initDownLSPre = dataSegment(cluIndsLSPre==2);

    else

        initUpLSPre = dataSegment(cluIndsLSPre==2); initDownLSPre = dataSegment(cluIndsLSPre==1);

    end


    % Turn lambda into lambda probs
    l1LSPre = mean(initUpLSPre); l2LSPre = mean(initDownLSPre); lambdaLSPre = [l1LSPre l2LSPre]./(l1LSPre+l2LSPre);

    % Initialise TR matrix
    initTransMat = rand(params.nStates);

    % Initialise EM matrix
    initEmisLSPre = [lambdaLSPre(2) lambdaLSPre(1); lambdaLSPre(1) lambdaLSPre(2)];

    % Initial sequence guess
    initSeqLSPre = ones(1,length(dataSegment));
    initSeqLSPre(dataSegment>=nanmean(dataSegment))=2;

    % Train the HMM using the Baum-Welch algorithm
    [estTRLSPre{iChunk},estEMLSPre{iChunk}] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaLSPre(:,:,iChunk),lllLSPre(iChunk)] = hmmdecode(initSeqLSPre, estTRLSPre{iChunk}, estEMLSPre{iChunk});

    % Get goodness of fit using AIC and BIC
    [~, AICLSPre(iChunk), BICLSPre(iChunk)] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre{iChunk}, estEMLSPre{iChunk}, lllLSPre(iChunk));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqLSPre(:,iChunk)] = hmmviterbi(initSeqLSPre, estTRLSPre{iChunk}, estEMLSPre{iChunk});

    % Fix small bursts using a history of 6 bins
    stateSeqLSPre(:,iChunk) = smoothOverChunks(stateSeqLSPre(:,iChunk)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateLSPre{iChunk}, dursUpStateLSPre{iChunk}] = getBinaryChunkDurations(stateSeqLSPre(:,iChunk));
    dursDownStateLSPre{iChunk} = dursDownStateLSPre{iChunk}./params.binnedFs; dursUpStateLSPre{iChunk} = dursUpStateLSPre{iChunk}./params.binnedFs;

    clear initSeqLSPre; clear initUpLSPre; clear initDownLSPre; clear initEmisLSPre;

end

% LSPost

for iChunk = 1:size(dataLSPost,2)

    % Get the segment out
    dataSegment = dataLSPost(:,iChunk);
    dataSegment(dataSegment==0) = 0.0001;
    % Run k-means to get initial clusters
    [cluIndsLSPost] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','iter');
    
     tmp1 = dataSegment(cluIndsLSPost==1); tmp2 = dataSegment(cluIndsLSPost==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPost = dataSegment(cluIndsLSPost==1); initDownLSPost = dataSegment(cluIndsLSPost==2);

    else

        initUpLSPost = dataSegment(cluIndsLSPost==2); initDownLSPost = dataSegment(cluIndsLSPost==1);

    end


    % Turn lambda into lambda probs
    l1LSPost = mean(initUpLSPost); l2LSPost = mean(initDownLSPost); lambdaLSPost = [l1LSPost l2LSPost]./(l1LSPost+l2LSPost);

    % Initialise TR matrix
    initTransMat = rand(params.nStates);

    % Initialise EM matrix
    initEmisLSPost = [lambdaLSPost(2) lambdaLSPost(1); lambdaLSPost(1) lambdaLSPost(2)];

    % Initial sequence guess
    initSeqLSPost = ones(1,length(dataSegment));
    initSeqLSPost(dataSegment>=nanmean(dataSegment))=2;

    % Train the HMM using the Baum-Welch algorithm
    [estTRLSPost{iChunk},estEMLSPost{iChunk}] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaLSPost(:,:,iChunk),lllLSPost(iChunk)] = hmmdecode(initSeqLSPost, estTRLSPost{iChunk}, estEMLSPost{iChunk});

    % Get goodness of fit using AIC and BIC
    [~, AICLSPost(iChunk), BICLSPost(iChunk)] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost{iChunk}, estEMLSPost{iChunk}, lllLSPost(iChunk));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqLSPost(:,iChunk)] = hmmviterbi(initSeqLSPost, estTRLSPost{iChunk}, estEMLSPost{iChunk});

    % Fix small bursts using a history of 6 bins
    stateSeqLSPost(:,iChunk) = smoothOverChunks(stateSeqLSPost(:,iChunk)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateLSPost{iChunk}, dursUpStateLSPost{iChunk}] = getBinaryChunkDurations(stateSeqLSPost(:,iChunk));
    dursDownStateLSPost{iChunk} = dursDownStateLSPost{iChunk}./params.binnedFs; dursUpStateLSPost{iChunk} = dursUpStateLSPost{iChunk}./params.binnedFs;

    clear initSeqLSPost; clear initUpLSPost; clear initDownLSPost; clear initEmisLSPost;

end


for iChunk = 1:size(alphaAwake,3)
    aPUAwake(iChunk) = nanmean(alphaAwake(2,:,iChunk));
end
for iChunk = 1:size(alphaLSPre,3)
    aPULSPre(iChunk) = nanmean(alphaLSPre(2,:,iChunk));
end
for iChunk = 1:size(alphaLSPost,3)
    aPULSPost(iChunk) = nanmean(alphaLSPost(2,:,iChunk));
end

%% Plot the shit
figure;
yyaxis left
h1=plot((2:length(BICAwake)-1)*params.chunkSize,(BICAwake(2:end-1)),'LineWidth',1.5);
hold on
h2=plot((2:length(BICLSPre)-1)*params.chunkSize,(BICLSPre(2:end-1)),'LineWidth',1.5);
h3=plot((2:length(BICLSPost)-1)*params.chunkSize,(BICLSPost(2:end-1)),'LineWidth',1.5);
box off
xlabel('time bins [s]')
ylabel('model (in)fidelity (AIC)')
title(['HMM fits for Session - '  dataset'])

yyaxis right
h4=plot((2:length(aPUAwake)-1)*params.chunkSize,(aPUAwake(2:end-1)),'LineWidth',2.5);
hold on
h5=plot((2:length(aPULSPre)-1)*params.chunkSize,(aPULSPre(2:end-1)),'LineWidth',2.5);
h6=plot((2:length(aPULSPost)-1)*params.chunkSize,(aPULSPost(2:end-1)),'LineWidth',2.5);
box off
xlabel('time bins [s]')
ylabel('P(UP)')
legend([h1,h2,h3,h4,h5,h6], 'GOF QW','GOF LS PreStim RS','GOF LS PostStim RS','P(UP) QW', 'P(UP) LS PreStim RS', 'P(UP) LS PostStim RS')


