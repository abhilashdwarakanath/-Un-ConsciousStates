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
params.binWidth = 0.05; %s
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't whine.
if str2double(dataset)==1
    params.firingThresh = 0.25; %Hz
elseif str2double(dataset)==4
    params.firingThresh = 7.5; %Hz
else
    params.firingThresh = 5;
end

params.binnedFs = 1/params.binWidth;
params.chunkSize = 20; %s
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
idx1 = firingRatesAwake >= params.firingThresh;
nValChansAwake = sum(idx1);

firingRatesLSPre = sum(rasterDataLSPre,2)/tLSPre(end);
idx2 = firingRatesAwake >= params.firingThresh;
nValChansLSPre = sum(idx2);

firingRatesLSPost = sum(rasterDataLSPost,2)/tLSPost(end);
idx3 = firingRatesLSPost >= params.firingThresh;
nValChansLSPost = sum(idx3);

valRasterAwake = rasterDataAwake(idx1,:);
valRasterLSPre = rasterDataLSPre(idx2,:);
valRasterLSPost = rasterDataLSPost(idx3,:);

% Collect spike counts by binning

binsAwake = 0:params.binWidth:tAwake(end);
binsLSPre = 0:params.binWidth:tLSPre(end);
binsLSPost = 0:params.binWidth:tLSPost(end);

spikeCountsAwake = zeros(nValChansAwake,length(binsAwake));
spikeCountsLSPre = zeros(nValChansLSPre,length(binsLSPre));
spikeCountsLSPost = zeros(nValChansLSPost,length(binsLSPost));

spkTimesAwake = cell(1,nValChansAwake);
spkTimesLSPre = cell(1,nValChansLSPre);
spkTimesLSPost = cell(1,nValChansLSPost);

for iChan = 1:nValChansAwake

    idx = find(valRasterAwake(iChan,:)==1);
    spkTimesAwake{iChan} = tAwake(idx);
    spikeCountsAwake(iChan,:) = histc(spkTimesAwake{iChan},binsAwake)./params.binWidth;
    clear idx;

end

for iChan = 1:nValChansLSPre

    idx = find(valRasterLSPre(iChan,:)==1);
    spkTimesLSPre{iChan} = tLSPre(idx);
    spikeCountsLSPre(iChan,:) = histc(spkTimesLSPre{iChan},binsLSPre)./params.binWidth;
    clear idx;

end

for iChan = 1:nValChansLSPost

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

for iSorted = 1:nValChansAwake

    sortedSpkTimesAwake{iSorted} = spkTimesAwake{idx1(iSorted)};

end

for iSorted = 1:nValChansLSPre

    sortedSpkTimesLSPre{iSorted} = spkTimesLSPre{idx2(iSorted)};

end

for iSorted = 1:nValChansLSPost

    sortedSpkTimesLSPost{iSorted} = spkTimesLSPost{idx3(iSorted)};

end

spkTimesAwake = sortedSpkTimesAwake;
spkTimesLSPre = sortedSpkTimesLSPre;
spkTimesLSPost = sortedSpkTimesLSPost;

%% Do the fitting shit

% Awake

for iChan = 1:nValChansAwake

    % Get the segment out
    dataSegment = spikeCountsAwake(iChan,:);
    dataSegment = normalise(dataSegment');
    dataSegment(dataSegment==0) = 0.0001;


    % Run k-means to get initial clusters
    [cluIndsAwake] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

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
    [estTRAwake{iChan},estEMAwake{iChan}] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaAwake(:,:,iChan),lllAwake(iChan)] = hmmdecode(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan});

    % Get goodness of fit using AIC and BIC
    [~, AICAwake(iChan), BICAwake(iChan)] = compute_goodness_of_fit(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan}, lllAwake(iChan));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqAwake(:,iChan)] = hmmviterbi(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan});

    % Fix small bursts using a history of 6 bins
    stateSeqAwake(:,iChan) = smoothOverChunks(stateSeqAwake(:,iChan)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateAwake{iChan}, dursUpStateAwake{iChan}] = getBinaryChunkDurations(stateSeqAwake(:,iChan));
    dursDownStateAwake{iChan} = dursDownStateAwake{iChan}./params.binnedFs; dursUpStateAwake{iChan} = dursUpStateAwake{iChan}./params.binnedFs;

    clear initSeqAwake; clear initUpAwake; clear initDownAwake; clear initEmisAwake;

end


% LSPre

for iChan = 1:nValChansLSPre

    % Get the segment out and adjust for 0s to avoid NaNs, integer overflow etc. Ugly but works.
    dataSegment = spikeCountsLSPre(iChan,:);
    dataSegment = normalise(dataSegment');
    dataSegment(dataSegment==0) = 0.0001;

    % Run k-means to get initial clusters
    [cluIndsLSPre] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

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
    [estTRLSPre{iChan},estEMLSPre{iChan}] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaLSPre(:,:,iChan),lllLSPre(iChan)] = hmmdecode(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan});

    % Get goodness of fit using AIC and BIC
    [~, AICLSPre(iChan), BICLSPre(iChan)] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan}, lllLSPre(iChan));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqLSPre(:,iChan)] = hmmviterbi(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan});

    % Fix small bursts using a history of 6 bins
    stateSeqLSPre(:,iChan) = smoothOverChunks(stateSeqLSPre(:,iChan)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateLSPre{iChan}, dursUpStateLSPre{iChan}] = getBinaryChunkDurations(stateSeqLSPre(:,iChan));
    dursDownStateLSPre{iChan} = dursDownStateLSPre{iChan}./params.binnedFs; dursUpStateLSPre{iChan} = dursUpStateLSPre{iChan}./params.binnedFs;

    clear initSeqLSPre; clear initUpLSPre; clear initDownLSPre; clear initEmisLSPre;

end

% LSPost

for iChan = 1:nValChansLSPost

   % Get the segment out
    dataSegment = spikeCountsLSPost(iChan,:);
    dataSegment = normalise(dataSegment');
    dataSegment(dataSegment==0) = 0.0001;

    % Run k-means to get initial clusters
    [cluIndsLSPost] = kmeans(dataSegment, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

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
    [estTRLSPost{iChan},estEMLSPost{iChan}] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol, 'Maxiterations',params.nIter);

    % Get the path probability sequences and log-likelihood
    [alphaLSPost(:,:,iChan),lllLSPost(iChan)] = hmmdecode(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan});

    % Get goodness of fit using AIC and BIC
    [~, AICLSPost(iChan), BICLSPost(iChan)] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan}, lllLSPost(iChan));

    % Compute the final state sequence using the Viterbi algorithm
    [stateSeqLSPost(:,iChan)] = hmmviterbi(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan});

    % Fix small bursts using a history of 6 bins
    stateSeqLSPost(:,iChan) = smoothOverChunks(stateSeqLSPost(:,iChan)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateLSPost{iChan}, dursUpStateLSPost{iChan}] = getBinaryChunkDurations(stateSeqLSPost(:,iChan));
    dursDownStateLSPost{iChan} = dursDownStateLSPost{iChan}./params.binnedFs; dursUpStateLSPost{iChan} = dursUpStateLSPost{iChan}./params.binnedFs;

    clear initSeqLSPost; clear initUpLSPost; clear initDownLSPost; clear initEmisLSPost;

end

%% Collect and plot and do some statistics

clear g
data = [AICAwake'; AICLSPre'; AICLSPost'];
group = [ones(size(AICAwake')); 2*ones(size(AICLSPre')); 3*ones(size(AICLSPost'))];
colours = [ones(size(AICAwake')); 2*ones(size(AICLSPre')); 3*ones(size(AICLSPost'))];
g=gramm('x',group,'y',data,'color',colours);
g.stat_boxplot();
g.set_title(['Session -' dataset]);
g.draw()
