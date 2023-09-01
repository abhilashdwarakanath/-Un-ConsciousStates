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

% Remove chutiya neurons - FIX POPULATION TO AWAKE AS STANDARD

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

frAwake = frAwake; % Don't shuffle awake
frLSPre = shuffle2(frLSPre);
frLSPost = shuffle2(frLSPost);

%% Now fit the shit in MATLAB

[cluIndsAwake] = kmeans((frAwake'), params.nStates, 'Replicates', params.nIter/100, 'Display','final');
[cluIndsLSPre] = kmeans((frLSPre'), params.nStates, 'Replicates', params.nIter/100, 'Display','final');
[cluIndsLSPost] = kmeans((frLSPost'), params.nStates, 'Replicates', params.nIter/100, 'Display','final');

tmp1 = frAwake(cluIndsAwake==1); tmp2 = frAwake(cluIndsAwake==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpAwake = frAwake(cluIndsAwake==1); initDownAwake = frAwake(cluIndsAwake==2);

    else

        initUpAwake = frAwake(cluIndsAwake==2); initDownAwake = frAwake(cluIndsAwake==1);

    end

tmp1 = frLSPre(cluIndsLSPre==1); tmp2 = frLSPre(cluIndsLSPre==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPre = frLSPre(cluIndsLSPre==1); initDownLSPre = frLSPre(cluIndsLSPre==2);

    else

        initUpLSPre = frLSPre(cluIndsLSPre==2); initDownLSPre = frLSPre(cluIndsLSPre==1);

    end

tmp1 = frLSPost(cluIndsLSPost==1); tmp2 = frLSPost(cluIndsLSPost==2);

    %Assign states correctly

    if mean(tmp1) > mean(tmp2)

        initUpLSPost = frLSPost(cluIndsLSPost==1); initDownLSPost = frLSPost(cluIndsLSPost==2);

    else

        initUpLSPost = frLSPost(cluIndsLSPost==2); initDownLSPost = frLSPost(cluIndsLSPost==1);

    end

l1Awake = mean(initUpAwake); l2Awake = mean(initDownAwake); lambdaAwake = [l1Awake l2Awake]./(l1Awake+l2Awake);
l1LSPre = mean(initUpLSPre); l2LSPre = mean(initDownLSPre); lambdaLSPre = [l1LSPre l2LSPre]./(l1LSPre+l2LSPre);
l1LSPost = mean(initUpLSPost); l2LSPost = mean(initDownLSPost); lambdaLSPost = [l1LSPost l2LSPost]./(l1LSPost+l2LSPost);

initTransMat = rand(params.nStates);

initEmisAwake = [lambdaAwake(2) lambdaAwake(1); lambdaAwake(1) lambdaAwake(2)];
initEmisLSPre = [lambdaLSPre(2) lambdaLSPre(1); lambdaLSPre(1) lambdaLSPre(2)];
initEmisLSPost = [lambdaLSPost(2) lambdaLSPost(1); lambdaLSPost(1) lambdaLSPost(2)];

initSeqAwake = ones(1,length(frAwake));
initSeqAwake(normalise(frAwake)>=mean(normalise(frAwake)))=2;

initSeqLSPre = ones(1,length(frLSPre));
initSeqLSPre(normalise(frLSPre)>=mean(normalise(frLSPre)))=2;

initSeqLSPost = ones(1,length(frLSPost));
initSeqLSPost(normalise(frLSPost)>=mean(normalise(frLSPost)))=2;

rng(1234);
initSeqLSPre = shuffle2(initSeqLSPre); initSeqLSPost = shuffle2(initSeqLSPost);
[estTRAwake,estEMAwake] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter);
[estTRLSPre,estEMLSPre] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter);
[estTRLSPost,estEMLSPost] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol,'Maxiterations',params.nIter);

% get the path probability sequences and most likely state sequence
[alphaAwake,lllAwake] = hmmdecode(initSeqAwake, estTRAwake, estEMAwake);
[stateSeqAwake] = hmmviterbi(initSeqAwake, estTRAwake, estEMAwake);

[alphaLSPre,lllLSPre] = hmmdecode(initSeqLSPre, estTRLSPre, estEMLSPre);
[stateSeqLSPre] = hmmviterbi(initSeqLSPre, estTRLSPre, estEMLSPre);

[alphaLSPost,lllLSPost] = hmmdecode(initSeqLSPost, estTRLSPost, estEMLSPost);
[stateSeqLSPost] = hmmviterbi(initSeqLSPost, estTRLSPost, estEMLSPost);

stateSeqAwake = stateSeqAwake-1;
stateSeqLSPre = stateSeqLSPre-1;
stateSeqLSPost = stateSeqLSPost-1;

% Fix very small chunks
stateSeqAwake = smoothOverChunks(stateSeqAwake,params.history,params.binnedFs);
stateSeqLSPre = smoothOverChunks(stateSeqLSPre,params.history,params.binnedFs);
stateSeqLSPost = smoothOverChunks(stateSeqLSPost,params.history,params.binnedFs);

% Get durations
[dursDownStateAwake, dursUpStateAwake] = getBinaryChunkDurations(stateSeqAwake);
dursUpStateAwake = dursUpStateAwake./params.binnedFs; dursDownStateAwake = dursDownStateAwake./params.binnedFs;

[dursDownStateLSPre, dursUpStateLSPre] = getBinaryChunkDurations(stateSeqLSPre);
dursUpStateLSPre = dursUpStateLSPre./params.binnedFs; dursDownStateLSPre = dursDownStateLSPre./params.binnedFs;

[dursDownStateLSPost, dursUpStateLSPost] = getBinaryChunkDurations(stateSeqLSPost);
dursUpStateLSPost = dursUpStateLSPost./params.binnedFs; dursDownStateLSPost = dursDownStateLSPost./params.binnedFs;

%% Do statistics

[~, AICAwake, BICAwake] = compute_goodness_of_fit(initSeqAwake, estTRAwake, estEMAwake, lllAwake);
[~, AICLSPre, BICLSPre] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre, estEMLSPre, lllLSPre);
[~, AICLSPost, BICLSPost] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost, estEMLSPost, lllLSPost);

%% Plot the shit

figure;
sgtitle('Quiet Wakefulness')
subplot(2,1,1)
yyaxis left
for i = 1:length(spkTimesAwake)
    st = spkTimesAwake{i}(spkTimesAwake{i}<20);
    plot([st; st], [i i+0.5],'-k')
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesAwake)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsAwake, smooth(alphaAwake(1,:)),'LineWidth',1.5)
hold on
plot(binsAwake,stateSeqAwake,'-g','LineWidth',1.5)
%p = patch(binsAwake, smooth((frAwake)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Pre-stimulation Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursUpStateAwake,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursDownStateAwake, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('UP', 'DOWN')
xlabel('duration [s]'); axis tight;
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;


figure;
sgtitle('Light Anaesthesia - Pre Stimulation')
subplot(2,1,1)
yyaxis left
for i = 1:length(spkTimesLSPre)
    st = spkTimesLSPre{i}(spkTimesLSPre{i}<20);
    if ~isempty(st)
    plot([st; st], [i i+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesLSPre)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsLSPre, smooth(alphaLSPre(1,:)),'LineWidth',1.5)
hold on
plot(binsLSPre,stateSeqLSPre,'-g','LineWidth',1.5)
%p = patch(binsLSPre, smooth((frLSPre)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursUpStateLSPre,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursDownStateLSPre, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('UP', 'DOWN')
xlabel('duration [s]'); axis tight;
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;


figure;
sgtitle('Light Anaesthesia - Post Stimulation')
subplot(2,1,1)
yyaxis left
for i = 1:length(spkTimesLSPost)
    st = spkTimesLSPost{i}(spkTimesLSPost{i}<20);
    if ~isempty(st)
    plot([st; st], [i i+0.5],'-k')
    end
    hold on
end
xlim([0, 20])
ylim([0, length(spkTimesLSPost)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsLSPost, smooth(alphaLSPost(1,:)),'LineWidth',1.5)
hold on
plot(binsLSPost,stateSeqLSPost,'-g','LineWidth',1.5)
%p = patch(binsLSPost, smooth((frLSPost)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursUpStateLSPost,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursDownStateLSPost, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('UP', 'DOWN'); axis tight;
xlabel('duration [s]')
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;

figure;

bar([AICAwake AICLSPre AICLSPost; 0 0 0],'grouped')
box off
ylabel('Bayesian model fidelity (AIC)')