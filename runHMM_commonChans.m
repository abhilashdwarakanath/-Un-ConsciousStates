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

% Remove chutiya neurons by selecting a common subspace of neurons across the conditions

firingRatesAwake = sum(rasterDataAwake,2)/tAwake(end);
idx1 = find(firingRatesAwake >= params.firingThresh);

firingRatesLSPre = sum(rasterDataLSPre,2)/tLSPre(end);
idx2 = find(firingRatesAwake >= params.firingThresh);

firingRatesLSPost = sum(rasterDataLSPost,2)/tLSPost(end);
idx3 = find(firingRatesLSPost >= params.firingThresh);

idx = intersect(intersect(idx1,idx2),idx3);
nValChans = length(idx);
valChans = idx;

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

%% Now fit the shit in MATLAB

tic;
% Set RND seed
rng(1234);


for iChan = 1:nValChans

    % Get the segments out and do shit
    ds1 = spikeCountsAwake(iChan,:);
    ds1 = normalise(ds1');
    ds1(ds1==0) = 0.0001;

    ds2 = spikeCountsLSPre(iChan,:);
    ds2 = normalise(ds2');
    ds2(ds2==0) = 0.0001;

    ds3 = spikeCountsLSPost(iChan,:);
    ds3 = normalise(ds3');
    ds3(ds3==0) = 0.0001;

    [cluIndsAwake] = kmeans(ds1, params.nStates, 'Replicates', params.nIter/100, 'Display','final');
    [cluIndsLSPre] = kmeans(ds2, params.nStates, 'Replicates', params.nIter/100, 'Display','final');
    [cluIndsLSPost] = kmeans(ds3, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

    tmp1 = ds1(cluIndsAwake==1); tmp2 = ds1(cluIndsAwake==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpAwake = ds1(cluIndsAwake==1); initDownAwake = ds1(cluIndsAwake==2);

    else

        initUpAwake = ds1(cluIndsAwake==2); initDownAwake = ds1(cluIndsAwake==1);

    end


    tmp1 = ds2(cluIndsLSPre==1); tmp2 = ds2(cluIndsLSPre==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPre = ds2(cluIndsLSPre==1); initDownLSPre = ds2(cluIndsLSPre==2);

    else

        initUpLSPre = ds2(cluIndsLSPre==2); initDownLSPre = ds2(cluIndsLSPre==1);

    end

    tmp1 = ds3(cluIndsLSPost==1); tmp2 = ds3(cluIndsLSPost==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPost = ds3(cluIndsLSPost==1); initDownLSPost = ds3(cluIndsLSPost==2);

    else

        initUpLSPost = ds3(cluIndsLSPost==2); initDownLSPost = ds3(cluIndsLSPost==1);

    end

    l1Awake = mean(initUpAwake); l2Awake = mean(initDownAwake); lambdaAwake = [l1Awake l2Awake]./(l1Awake+l2Awake);
    l1LSPre = mean(initUpLSPre); l2LSPre = mean(initDownLSPre); lambdaLSPre = [l1LSPre l2LSPre]./(l1LSPre+l2LSPre);
    l1LSPost = mean(initUpLSPost); l2LSPost = mean(initDownLSPost); lambdaLSPost = [l1LSPost l2LSPost]./(l1LSPost+l2LSPost);

    initTransMat = rand(params.nStates);

    initEmisAwake = [lambdaAwake(2) lambdaAwake(1); lambdaAwake(1) lambdaAwake(2)];
    initEmisLSPre = [lambdaLSPre(2) lambdaLSPre(1); lambdaLSPre(1) lambdaLSPre(2)];
    initEmisLSPost = [lambdaLSPost(2) lambdaLSPost(1); lambdaLSPost(1) lambdaLSPost(2)];

    initSeqAwake = ones(1,length(ds1));
    initSeqAwake(ds1>=mean(ds1))=2;

    initSeqLSPre = ones(1,length(ds2));
    initSeqLSPre(ds2>=mean(ds2))=2;

    initSeqLSPost = ones(1,length(ds3));
    initSeqLSPost(normalise(ds3)>=mean(ds3))=2;

    rng(1234);
    [estTRAwake{iChan},estEMAwake{iChan}] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol);
    [estTRLSPre{iChan},estEMLSPre{iChan}] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol);
    [estTRLSPost{iChan},estEMLSPost{iChan}] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol);

    % Get the path probability sequences and most likely state sequence
    [alphaAwake(:,:,iChan),lllAwake(iChan)] = hmmdecode(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan});
    [~, AICAwake(iChan), BICAwake(iChan)] = compute_goodness_of_fit(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan}, lllAwake(iChan));
    [stateSeqAwake(iChan,:)] = hmmviterbi(initSeqAwake, estTRAwake{iChan}, estEMAwake{iChan});

    [alphaLSPre(:,:,iChan),lllLSPre(iChan)] = hmmdecode(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan});
    [~, AICLSPre(iChan), BICLSPre(iChan)] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan}, lllLSPre(iChan));
    [stateSeqLSPre(iChan,:)] = hmmviterbi(initSeqLSPre, estTRLSPre{iChan}, estEMLSPre{iChan});

    [alphaLSPost(:,:,iChan),lllLSPost(iChan)] = hmmdecode(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan});
    [~, AICLSPost(iChan), BICLSPost(iChan)] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan}, lllLSPost(iChan));
    [stateSeqLSPost(iChan,:)] = hmmviterbi(initSeqLSPost, estTRLSPost{iChan}, estEMLSPost{iChan});


    % Fix very small chunks
    stateSeqAwake(iChan,:) = smoothOverChunks(stateSeqAwake(iChan,:)-1,params.history,params.binnedFs);
    stateSeqLSPre(iChan,:) = smoothOverChunks(stateSeqLSPre(iChan,:)-1,params.history,params.binnedFs);
    stateSeqLSPost(iChan,:) = smoothOverChunks(stateSeqLSPost(iChan,:)-1,params.history,params.binnedFs);

    % Get durations
    [dursDownStateAwake{iChan}, dursUpStateAwake{iChan}] = getBinaryChunkDurations(stateSeqAwake(iChan,:));
    dursDownStateAwake{iChan} = dursDownStateAwake{iChan}./params.binnedFs; dursUpStateAwake{iChan} = dursUpStateAwake{iChan}./params.binnedFs;

    [dursDownStateLSPre{iChan}, dursUpStateLSPre{iChan}] = getBinaryChunkDurations(stateSeqLSPre(iChan,:));
    dursDownStateLSPre{iChan} = dursDownStateLSPre{iChan}./params.binnedFs; dursUpStateLSPre{iChan} = dursUpStateLSPre{iChan}./params.binnedFs;

    [dursDownStateLSPost{iChan}, dursUpStateLSPost{iChan}] = getBinaryChunkDurations(stateSeqLSPost(iChan,:));
    dursDownStateLSPost{iChan} = dursDownStateLSPost{iChan}./params.binnedFs; dursUpStateLSPost{iChan} = dursUpStateLSPost{iChan}./params.binnedFs;

end

toc;

%% Plot some shit

% Combine the data into a single matrix
data = [AICAwake' AICLSPre' AICLSPost'];

% Plot the bar chart
figure;
bar(data, 'grouped');
legend('Awake RS', 'LS PreStim RS', 'LSPostStim RS');
xlabel('Channels');
ylabel('model (in)fidelity (AIC)');
title('2-HMM Goodness of Fit');
xticks(1:nValChans);
set(gca, 'XTickLabel', num2str(valChans));
box off
