clear all
clc
close all

dbstop if error

%% load dataset and set params

cd('C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\1\11')
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd('C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\1\21')
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd('C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\1\22')
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.05; %s
params.firingThresh = 0.25; %Hz
%params.firingThresh = 5; %Hz
params.binnedFs = 1/params.binWidth;
params.chunkDuration = 20; %s

params.nStates = 2;
params.nIter = 1e3;
params.tol = 1e-6;
params.smoothFac = 0.125; %Seconds

%% Do shit

% Remove chutiya neurons

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

%% Chunk them into T second chunks

dataAwake = chunkData(spikeCountsAwake,params.binnedFs,params.chunkDuration);
dataLSPre = chunkData(spikeCountsLSPre,params.binnedFs,params.chunkDuration);
dataLSPost = chunkData(spikeCountsLSPost,params.binnedFs,params.chunkDuration);

%% Now fit the shit in MATLAB

tic;

% Set RND seed
rng(1234);

for iChan = 1:nValChans

    for iChunk = 1:size(dataAwake,3)

    % Get the segments out and do shit % WRITE SEPARATE LOOPS FOR EACH CONDITION
    ds1 = dataAwake(iChan,:,iChunk);
    if sum(ds1)~=0
        ds1 = normalise(ds1);
    end
    ds1(ds1==0) = 0.0001; ds1 = ds1';

    ds2 = dataLSPre(iChan,:,iChunk);
    if sum(ds2)~= 0
        ds2 = normalise(ds2);
    end
    ds2(ds2==0) = 0.0001; ds2 = ds2';

    ds3 = dataLSPost(iChan,:,iChunk);
    if sum(ds3)~= 0
        ds3 = normalise(ds3);
    end
    ds3(ds3==0) = 0.0001; ds3 = ds3';

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

    [estTRAwake{iChan,iChunk},estEMAwake{iChan,iChunk}] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter,'Algorithm','BaumWelch');
    [estTRLSPre{iChan,iChunk},estEMLSPre{iChan,iChunk}] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter,'Algorithm','BaumWelch');
    [estTRLSPost{iChan,iChunk},estEMLSPost{iChan,iChunk}] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter,'Algorithm','BaumWelch');

    % Get the path probability sequences and most likely state sequence
    [alphaAwake(:,:,iChan,iChunk),lllAwake(iChan,iChunk)] = hmmdecode(initSeqAwake, estTRAwake{iChan,iChunk}, estEMAwake{iChan,iChunk});
    [~, AICAwake(iChan,iChunk), BICAwake(iChan,iChunk)] = compute_goodness_of_fit(initSeqAwake, estTRAwake{iChan,iChunk}, estEMAwake{iChan,iChunk}, lllAwake(iChan));
    [stateSeqAwake(iChan,:,iChunk)] = hmmviterbi(initSeqAwake, estTRAwake{iChan,iChunk}, estEMAwake{iChan,iChunk});

    [alphaLSPre(:,:,iChan,iChunk),lllLSPre(iChan,iChunk)] = hmmdecode(initSeqLSPre, estTRLSPre{iChan,iChunk}, estEMLSPre{iChan,iChunk});
    [~, AICLSPre(iChan,iChunk), BICLSPre(iChan,iChunk)] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre{iChan,iChunk}, estEMLSPre{iChan,iChunk}, lllLSPre(iChan));
    [stateSeqLSPre(iChan,:,iChunk)] = hmmviterbi(initSeqLSPre, estTRLSPre{iChan,iChunk}, estEMLSPre{iChan,iChunk});

    [alphaLSPost(:,:,iChan,iChunk),lllLSPost(iChan,iChunk)] = hmmdecode(initSeqLSPost, estTRLSPost{iChan,iChunk}, estEMLSPost{iChan,iChunk});
    [~, AICLSPost(iChan,iChunk), BICLSPost(iChan,iChunk)] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost{iChan,iChunk}, estEMLSPost{iChan,iChunk}, lllLSPost(iChan));
    [stateSeqLSPost(iChan,:,iChunk)] = hmmviterbi(initSeqLSPost, estTRLSPost{iChan,iChunk}, estEMLSPost{iChan,iChunk});


    % Fix very small chunks
    stateSeqAwake(iChan,:,iChunk) = smoothOverChunks(stateSeqAwake(iChan,:,iChunk)-1,params.smoothFac,params.binnedFs);
    stateSeqLSPre(iChan,:,iChunk) = smoothOverChunks(stateSeqLSPre(iChan,:,iChunk)-1,params.smoothFac,params.binnedFs);
    stateSeqLSPost(iChan,:,iChunk) = smoothOverChunks(stateSeqLSPost(iChan,:,iChunk)-1,params.smoothFac,params.binnedFs);

    % Get durations
    [dursState1Awake{iChan,iChunk}, dursState2Awake{iChan,iChunk}] = getBinaryChunkDurations(stateSeqAwake(iChan,:));
    dursState1Awake{iChan,iChunk} = dursState1Awake{iChan,iChunk}./params.binnedFs; dursState2Awake{iChan,iChunk} = dursState2Awake{iChan,iChunk}./params.binnedFs;

    [dursState1LSPre{iChan,iChunk}, dursState2LSPre{iChan,iChunk}] = getBinaryChunkDurations(stateSeqLSPre(iChan,:));
    dursState1LSPre{iChan,iChunk} = dursState1LSPre{iChan,iChunk}./params.binnedFs; dursState2LSPre{iChan,iChunk} = dursState2LSPre{iChan,iChunk}./params.binnedFs;

    [dursState1LSPost{iChan,iChunk}, dursState2LSPost{iChan,iChunk}] = getBinaryChunkDurations(stateSeqLSPost(iChan,:));
    dursState1LSPost{iChan,iChunk} = dursState1LSPost{iChan,iChunk}./params.binnedFs; dursState2LSPost{iChan,iChunk} = dursState2LSPost{iChan,iChunk}./params.binnedFs;

    end

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
ylabel('AIC []');
title('2-HMM Goodness of Fit');
xticks(1:nValChans);
set(gca, 'XTickLabel', num2str(valChans));
