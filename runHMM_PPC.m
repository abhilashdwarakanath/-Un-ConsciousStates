clear all
clc
close all

dbstop if error

%% Ask user for input for the session #1

userInput = inputdlg('Enter Session number:', 'Integer Input', [1 40]);
session = str2double(userInput);

if ~isempty(session) && isnumeric(session) && session >= 1 && session <= 4
    postFAHA = 2;
    h=msgbox(['Analysing Session: ' num2str(session)], 'Valid Input');
    pause(3);
    delete(h);
else
    h=msgbox('Session not parsed yet.', 'Invalid Input');
    pause(3);
    delete(h);
end

%% load dataset and set params

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PPC\' num2str(session) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PPC\' num2str(session) '\21'])
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PPC\' num2str(session) '\' '2' num2str(postFAHA)])
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.025; %s
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't whine.
params.firingThresh = 0.05; %Hz
params.binnedFs = 1/params.binWidth;
params.chunkSize = 20; %s
params.channelThreshold = 15;
params.offset = 2;
params.nStates = 2;
params.nIter = 1e3;
params.tol = 1e-6;
params.history = 0.0250; %Seconds

%% If session = 4, clean the dataset again

if session == 4

    rasterDataAwake = removeCommonArtifacts(params,rasterDataAwake);
    rasterDataLSPre = removeCommonArtifacts(params,rasterDataLSPre);
    rasterDataLSPost = removeCommonArtifacts(params,rasterDataLSPost);

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

%% Now fit the shit in MATLAB

[cluIndsAwake] = kmeans(frAwake', params.nStates, 'Replicates', params.nIter/100, 'Display','final');
[cluIndsLSPre] = kmeans(frLSPre', params.nStates, 'Replicates', params.nIter/100, 'Display','final');
[cluIndsLSPost] = kmeans(frLSPost', params.nStates, 'Replicates', params.nIter/100, 'Display','final');

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

initTransMat = 0.3+(0.7-0.3)*rand(params.nStates);

initEmisAwake = [lambdaAwake(2) lambdaAwake(1); lambdaAwake(1) lambdaAwake(2)];
initEmisLSPre = [lambdaLSPre(2) lambdaLSPre(1); lambdaLSPre(1) lambdaLSPre(2)];
initEmisLSPost = [lambdaLSPost(2) lambdaLSPost(1); lambdaLSPost(1) lambdaLSPost(2)];

initSeqAwake = ones(1,length(frAwake));
initSeqAwake(frAwake>=mean(frAwake))=2;

initSeqLSPre = ones(1,length(frLSPre));
initSeqLSPre(frLSPre>=mean(frLSPre))=2;

initSeqLSPost = ones(1,length(frLSPost));
initSeqLSPost(frLSPost>=mean(frLSPost))=2;

rng(1234);
[estTRAwake,estEMAwake] = hmmtrain(initSeqAwake,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol);
[estTRLSPre,estEMLSPre] = hmmtrain(initSeqLSPre,initTransMat,initEmisLSPre,'Verbose',true,'Tolerance',params.tol);
[estTRLSPost,estEMLSPost] = hmmtrain(initSeqLSPost,initTransMat,initEmisLSPost,'Verbose',true,'Tolerance',params.tol);

% Get the path probability sequences and most likely state sequence
[alphaAwake,lllAwake] = hmmdecode(initSeqAwake, estTRAwake, estEMAwake);
[~, AICAwake, BICAwake] = compute_goodness_of_fit(initSeqAwake, estTRAwake, estEMAwake, lllAwake);
[stateSeqAwake] = hmmviterbi(initSeqAwake, estTRAwake, estEMAwake);

[alphaLSPre,lllLSPre] = hmmdecode(initSeqLSPre, estTRLSPre, estEMLSPre);
[~, AICLSPre, BICLSPre] = compute_goodness_of_fit(initSeqLSPre, estTRLSPre, estEMLSPre, lllLSPre);
[stateSeqLSPre] = hmmviterbi(initSeqLSPre, estTRLSPre, estEMLSPre);

[alphaLSPost,lllLSPost] = hmmdecode(initSeqLSPost, estTRLSPost, estEMLSPost);
[~, AICLSPost, BICLSPost] = compute_goodness_of_fit(initSeqLSPost, estTRLSPost, estEMLSPost, lllLSPost);
[stateSeqLSPost] = hmmviterbi(initSeqLSPost, estTRLSPost, estEMLSPost);

% Fix very small chunks
stateSeqAwake = smoothOverChunks(stateSeqAwake-1,params.history,params.binnedFs);
stateSeqLSPre = smoothOverChunks(stateSeqLSPre-1,params.history,params.binnedFs);
stateSeqLSPost = smoothOverChunks(stateSeqLSPost-1,params.history,params.binnedFs);

% Get durations
[dursDownStateAwake, dursUpStateAwake] = getBinaryChunkDurations(stateSeqAwake);
dursDownStateAwake = dursDownStateAwake./params.binnedFs; dursUpStateAwake = dursUpStateAwake./params.binnedFs;

[dursDownStateLSPre, dursUpStateLSPre] = getBinaryChunkDurations(stateSeqLSPre);
dursDownStateLSPre = dursDownStateLSPre./params.binnedFs; dursUpStateLSPre = dursUpStateLSPre./params.binnedFs;

[dursDownStateLSPost, dursUpStateLSPost] = getBinaryChunkDurations(stateSeqLSPost);
dursDownStateLSPost = dursDownStateLSPost./params.binnedFs; dursUpStateLSPost = dursUpStateLSPost./params.binnedFs;

%% Do some statistics

% Get UP and DOWN state frequencies

freqsUpStateLSPre = 1./dursUpStateLSPre; freqsDownStateLSPre = 1./dursDownStateLSPre;
freqsUpStateLSPost = 1./dursUpStateLSPost; freqsDownStateLSPost = 1./dursDownStateLSPost;
binSize = 1 / (100*params.history);
binsForFitting = 0:binSize:(0.25/params.history); % Max frequency corresponds to smallest possible state duration

clear h
tic;

[upDursLSPre,~] = ksdensity(freqsUpStateLSPre,binsForFitting,'Support','positive','BoundaryCorrection','reflection'); [downDursLSPre,~] = ksdensity(freqsDownStateLSPre,binsForFitting,'Support','positive','BoundaryCorrection','reflection');
[upDursLSPost,~] = ksdensity(freqsUpStateLSPost,binsForFitting,'Support','positive','BoundaryCorrection','reflection'); [downDursLSPost,~] = ksdensity(freqsDownStateLSPost,binsForFitting,'Support','positive','BoundaryCorrection','reflection');
toc;

[p(1),h(1)] = ranksum(dursUpStateLSPre,dursUpStateLSPost);
[p(2),h(2)] = ranksum(dursDownStateLSPre,dursDownStateLSPost);

% Get cycle alternation frequency

nCyclesLSPre = min(length(dursUpStateLSPre), length(dursDownStateLSPre));
cycleDursLSPre = 1./(dursUpStateLSPre(1:nCyclesLSPre) + dursDownStateLSPre(1:nCyclesLSPre));

nCyclesLSPost = min(length(dursUpStateLSPost), length(dursDownStateLSPost));
cycleDursLSPost = 1./(dursUpStateLSPost(1:nCyclesLSPost) + dursDownStateLSPost(1:nCyclesLSPost));

cycleFreqsLSPre = ksdensity(cycleDursLSPre,binsForFitting);
cycleFreqsLSPost = ksdensity(cycleDursLSPost,binsForFitting);

[p(3),h(3)] = ranksum(cycleDursLSPre,cycleDursLSPost);

figure('units','normalized','outerposition',[0 0 1 1]);
bar([AICAwake AICLSPre AICLSPost; NaN NaN NaN])
ax = gca;
ax.XLim = [0.5 1.5];
box off
ylabel('model (in)fidelity - AIC []')
legend('Quiet Wakefulness','Light Anaesthesia - PreStim', 'Light Anaesthesia - PostStim')

figure('units','normalized','outerposition',[0 0 1 1]);
plot(binsForFitting,(cycleFreqsLSPre./length(cycleFreqsLSPre)),'LineWidth',2)
hold on
plot(binsForFitting,(cycleFreqsLSPost./length(cycleFreqsLSPost)),'LineWidth',2)
box off
vline(median(cycleDursLSPre),'--b');vline(median(cycleDursLSPost),'--r');
xlabel('UP/DOWN cycle frequency [Hz]')
ylabel('normalised density []')
legend('LS PreStim RS','LS PostStim RS')
title(['Cycle Frequency Change - Session # - ' num2str(session)])

figure('units','normalized','outerposition',[0 0 1 1]);
sgtitle(['Session # : ' num2str(session)])
subplot(1,2,1)
plot(binsForFitting,upDursLSPre./length(upDursLSPre),'LineWidth',2)
hold on
plot(binsForFitting,upDursLSPost./length(upDursLSPost),'LineWidth',2)
xlabel('alternation frequency [Hz]')
ylabel('normalised density []')
title('UP States'); box off;
legend('LS PreStim RS','LS PostStim RS')

subplot(1,2,2)
plot(binsForFitting,downDursLSPre./length(downDursLSPre),'LineWidth',2)
hold on
plot(binsForFitting,downDursLSPost./length(downDursLSPost),'LineWidth',2)
xlabel('alternation frequency [Hz]')
ylabel('normalised density []')
title('DOWN States'); box off;
legend('LS PreStim RS','LS PostStim RS')

%% Plot one over the other

figure;
subplot(3,1,1)
yyaxis left
for i = 1:length(spkTimesAwake)
    st = spkTimesAwake{i}(spkTimesAwake{i}<20);
    if ~isempty(st)
        plot([st; st], [i i+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesAwake)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsAwake, smooth(alphaAwake(1,:)),'LineWidth',1.5)
hold on
plot(binsAwake,stateSeqAwake,'-g','LineWidth',1.5)
ylabel('P(UP State)')
xlabel('time [s]')
title('Pre-stimulation Resting State')
xlim([0 20])
box off;

subplot(3,1,2)
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
ylabel('P(UP State)')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(3,1,3)
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
ylabel('P(UP State)')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

%% Plot the shit

figure('units','normalized','outerposition',[0 0 1 1]);
sgtitle('Quiet Wakefulness')
subplot(2,1,1)
yyaxis left
for i = 1:length(spkTimesAwake)
    st = spkTimesAwake{i};
    if ~isempty(st)
        plot([st; st], [i i+0.5],'-k')
    end
    hold on
end
xlim([0 20])
ylim([0, length(spkTimesAwake)+1])
ylabel('Neuron ID')
yyaxis right
plot(binsAwake, smooth(alphaAwake(1,:)),'LineWidth',1.5)
hold on
plot(binsAwake,stateSeqAwake,'-g','LineWidth',1.5)
%p = patch(binsAwake, smooth(normalise(frAwake)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Pre-stimulation Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateAwake,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateAwake, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;


figure('units','normalized','outerposition',[0 0 1 1]);
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
%p = patch(binsLSPre, smooth(normalise(frLSPre)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateLSPre,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateLSPre, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;


figure('units','normalized','outerposition',[0 0 1 1]);
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
%p = patch(binsLSPost, smooth(normalise(frLSPost)), 'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'spike density');
ylabel('P(UP State)')
%legend(p, {'spike density function [spk/s]'}, 'Location', 'northwest')
xlabel('time [s]')
title('Resting State')
xlim([0 20])
box off;

subplot(2,1,2)
histogram(dursDownStateLSPost,25, 'Normalization', 'pdf','FaceAlpha',0.5)
hold on
histogram(dursUpStateLSPost, 25, 'Normalization', 'pdf','FaceAlpha',0.25)
legend('DOWN', 'UP')
xlabel('duration [s]')
ylabel('state probability []')
title('Duration distribution of hidden states')
box off;

%% Save all the figures

cd('C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PPC\Results')
cd genWithFolds
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

pause(3)
close all
