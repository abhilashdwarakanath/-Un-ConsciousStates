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

if ~isempty(session) && isnumeric(session) && session ~= 3

	h=msgbox('Session not valid.', 'Invalid Input');
	pause(3);
	delete(h);

else

	h=msgbox(['Analysing Session: ' num2str(session)], 'Valid Input');
	pause(3);
	delete(h);

end

clear h;

%% load dataset and set params

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' num2str(session) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
lfpDataAwake = neuralActivity.lfp.activity;
tLFPAwake = neuralActivity.lfp.t;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\' num2str(session) '\21'])
load neuralActivity.mat

rasterDataLS = neuralActivity.spikes.activity;
lfpDataLS = neuralActivity.lfp.activity;
tLS = neuralActivity.spikes.t;

clear neuralActivity;

%% Set Params

params.Fs = 1/(tAwake(3)-tAwake(2)); % Hz
params.binWidth = 0.025;
params.elecs = 192; % JUST SO THAT removeCommonArtifacts() doesn't cry. This was when we had 2 arrays, 96x2.
if session == 1
	params.firingThresh = 0.05; %Hz. Atleast fire some spikes.
else
	params.firingThresh = 2.5;
end
params.binnedFs = 1/params.binWidth;
params.chunkSize = 10; %s. To plot representative rasters
params.channelThreshold = 15; % To remove cross-channel movement artifacts.
params.offset = 2; % Machine lag can occur.
params.nStates = 2; % For the HMM
params.nIter = 1e3; % max iterations for convergence
params.tol = 1e-6; % convergence threshold.
params.history = 0.075;
params.nFolds = 1; % For training the HMM on LS PreStim RS
params.holdOut = 0;% percent

%% Do shit

% Remove chutiya neurons

% FIXED POPULATION ANALYSIS. Select good firing neurons based on a firing rate threshold from the awake block. Use those neurons also in the anaesthetised
% blocks. This way we can track the change in population activity.

firingRatesAwake = sum(rasterDataAwake,2)/tAwake(end);
idx = firingRatesAwake >= params.firingThresh;
nValChans = sum(idx);

valRasterAwake = rasterDataAwake(idx,:);
valRasterLS = rasterDataLS(idx,:);

% Get the corresponding LFPs

valLFPAwake = lfpDataAwake(idx,:);
valLFPLS = lfpDataLS(idx,:);

clear lfpDataAwake; clear lfpDataLS;

% Collect spike counts by binning

binsAwake = 0:params.binWidth:tAwake(end);
binsLS = 0:params.binWidth:tLS(end);

spikeCountsAwake = zeros(nValChans,length(binsAwake));
spikeCountsLS = zeros(nValChans,length(binsLS));

spkTimesAwake = cell(1,nValChans);
spkTimesLS = cell(1,nValChans);

for iChan = 1:nValChans

	idx = valRasterAwake(iChan,:)==1;
	spkTimesAwake{iChan} = tAwake(idx);
	spikeCountsAwake(iChan,:) = histc(spkTimesAwake{iChan},binsAwake)./params.binWidth;
	clear idx;

	idx = valRasterLS(iChan,:)==1;
	spkTimesLS{iChan} = tLS(idx);
	spikeCountsLS(iChan,:) = histc(spkTimesLS{iChan},binsLS)./params.binWidth;
	clear idx;

end

% Create the PSTH and normalise it between 0 and 1. Then constrain it between 1e-3 and 0.999. Avoid overflow shit.

frAwake = normalise(nanmean(spikeCountsAwake,1));
frAwake(frAwake==0) = 1e-3; frAwake(frAwake==1) = 0.999;
frLS = normalise(nanmean(spikeCountsLS,1));
frLS(frLS==0) = 1e-3; frLS(frLS==1) = 0.999;

clear rasterDataAwake; clear rasterDataLS;

%% Partition and fit

% Generate a "random" initial transition matrix
initTransMat = 0.3+(0.7-0.3)*rand(params.nStates);

% Create the initial guess sequences
initSeqAwake = ones(1,length(frAwake));
initSeqAwake(frAwake>=mean(frAwake))=2;

initSeqLS = ones(1,length(frLS));
initSeqLS(frLS>=mean(frLS))=2;

rng(1234); % Random seed for reproducibility

% Awake -

nTrainingSamps = floor((1-params.holdOut)*length(frAwake));
trainingData = frAwake(1:nTrainingSamps);

%Fit in nFolds and take the average estimate

foldSize = floor(length(trainingData)/params.nFolds);
remainder = mod(length(trainingData), params.nFolds);
trainingData = trainingData(1:end-remainder);

foldedData = reshape(trainingData, foldSize, params.nFolds);
estTRAwake = zeros(params.nStates); estEMAwake = zeros(params.nStates);

for iFold = 1:params.nFolds

	data = foldedData(:,iFold);
	% Run 2 cluster k-means to extract the initial clusters
	[cluIndsAwake] = kmeans(data, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

	tmp1 = frAwake(cluIndsAwake==1); tmp2 = frAwake(cluIndsAwake==2);
	%Assign states correctly
	if mean(tmp1) > mean(tmp2)
		initUpAwake = frAwake(cluIndsAwake==1); initDownAwake = frAwake(cluIndsAwake==2);
	else
		initUpAwake = frAwake(cluIndsAwake==2); initDownAwake = frAwake(cluIndsAwake==1);
	end
	% Generate the initial emission matrix using normalised rates as the poisson lambdas
	l1Awake = mean(initUpAwake); l2Awake = mean(initDownAwake); lambdaAwake = [l1Awake l2Awake]./(l1Awake+l2Awake);
	initEmisAwake = [lambdaAwake(2) lambdaAwake(1); lambdaAwake(1) lambdaAwake(2)];
	initSeqAwakeTraining = ones(1,length(data));
	initSeqAwakeTraining(data>=mean(data))=2;
	% Train the HMM
	[tmp1,tmp2] = hmmtrain(initSeqAwakeTraining,initTransMat,initEmisAwake,'Verbose',true,'Tolerance',params.tol);
	estTRAwake = estTRAwake+tmp1; estEMAwake = estEMAwake+tmp2;

end

estTRAwake = estTRAwake./params.nFolds; estEMAwake = estEMAwake./params.nFolds;

% LS

nTrainingSamps = floor((1-params.holdOut)*length(frLS));
trainingData = frLS(1:nTrainingSamps);

%Fit in nFolds and take the average estimate

foldSize = floor(length(trainingData)/params.nFolds);
remainder = mod(length(trainingData), params.nFolds);
trainingData = trainingData(1:end-remainder);

foldedData = reshape(trainingData, foldSize, params.nFolds);
estTRLS = zeros(params.nStates); estEMLS = zeros(params.nStates);

for iFold = 1:params.nFolds

	data = foldedData(:,iFold);
	% Run 2 cluster k-means to extract the initial clusters
	[cluIndsLS] = kmeans(data, params.nStates, 'Replicates', params.nIter/100, 'Display','final');

	tmp1 = frLS(cluIndsLS==1); tmp2 = frLS(cluIndsLS==2);
	%Assign states correctly
	if mean(tmp1) > mean(tmp2)
		initUpLS = frLS(cluIndsLS==1); initDownLS = frLS(cluIndsLS==2);
	else
		initUpLS = frLS(cluIndsLS==2); initDownLS = frLS(cluIndsLS==1);
	end
	% Generate the initial emission matrix using normalised rates as the poisson lambdas
	l1LS = mean(initUpLS); l2LS = mean(initDownLS); lambdaLS = [l1LS l2LS]./(l1LS+l2LS);
	initEmisLS = [lambdaLS(2) lambdaLS(1); lambdaLS(1) lambdaLS(2)];
	initSeqLSTraining = ones(1,length(data));
	initSeqLSTraining(data>=mean(data))=2;
	% Train the HMM
	[tmp1,tmp2] = hmmtrain(initSeqLSTraining,initTransMat,initEmisLS,'Verbose',true,'Tolerance',params.tol);
	estTRLS = estTRLS+tmp1; estEMLS = estEMLS+tmp2;

end

estTRLS = estTRLS./params.nFolds; estEMLS = estEMLS./params.nFolds;

%% Do validation and Prediction

% Get the path probability sequences and most likely state sequence
[alphaAwake,lllAwake] = hmmdecode(initSeqAwake, estTRAwake, estEMAwake);
[~, AICAwake, BICAwake] = compute_goodness_of_fit(initSeqAwake, estTRAwake, estEMAwake, lllAwake);
[stateSeqAwake] = hmmviterbi(initSeqAwake, estTRAwake, estEMAwake);

[alphaLS,lllLS] = hmmdecode(initSeqLS, estTRLS, estEMLS);
[~, AICLS, BICLS] = compute_goodness_of_fit(initSeqLS, estTRLS, estEMLS, lllLS);
[stateSeqLS] = hmmviterbi(initSeqLS, estTRLS, estEMLS);

% Fix very small chunks using the "history" parameter
stateSeqAwake = smoothOverChunks(stateSeqAwake-1,params.history,params.binnedFs);
stateSeqLS = smoothOverChunks(stateSeqLS-1,params.history,params.binnedFs);

if session == 1 || session == 2

	stateSeqAwake = stateSeqAwake+1;
	stateSeqAwake(stateSeqAwake==2)=0;
	stateSeqLS = stateSeqLS+1;
	stateSeqLS(stateSeqLS==2)=0;

end

% Get state durations
[dursDownStateAwake, dursUpStateAwake] = getBinaryChunkDurations(stateSeqAwake);
dursDownStateAwake = dursDownStateAwake./params.binnedFs; dursUpStateAwake = dursUpStateAwake./params.binnedFs;

[dursDownStateLS, dursUpStateLS] = getBinaryChunkDurations(stateSeqLS);
dursDownStateLS = dursDownStateLS./params.binnedFs; dursUpStateLS = dursUpStateLS./params.binnedFs;

%% Plot ACFs and get synchrony index

if session == 1

	xlimits(:,1) = 0:10:1180;
	xlimits(:,2) = 10:10:1190;

elseif session == 3

	xlimits(:,1) = 0:10:1190;
	xlimits(:,2) = 10:10:1200;

else

	xlimits(:,1) = 0:10:560;
	xlimits(:,2) = 10:10:570;

end

for iEpoch = 1:size(xlimits,1)

	[~,idx1] = min(abs(binsAwake-xlimits(iEpoch,1)));
	[~,idx2] = min(abs(binsAwake-xlimits(iEpoch,2)));

	[acAwake(iEpoch,:),~,SIAwake(iEpoch)] = sig_autocorr(frAwake(idx1:idx2-1),params.binnedFs,params.history*25);
	[acLS(iEpoch,:),~,SILS(iEpoch)] = sig_autocorr(frLS(idx1:idx2-1),params.binnedFs,params.history*25);

end

% %Awake
% options.color_area = [0.0000, 0.4470, 0.7410];
% options.color_line = [0.0000, 0.4470, 0.7410];
% h = figure;
% hold on
% options.handle = h;
% options.alpha = 0.5;
% options.line_width = 2;
% options.x_axis = lags;
% options.error = 'c95';
% plot_areaerrorbar(acAwake, options)
% box off;
% 
% % LS
% h = figure;
% options.color_area = [0.8500, 0.3250, 0.0980];
% options.color_line = [0.8500, 0.3250, 0.0980];
% options.handle = h;
% options.alpha = 0.2;
% options.line_width = 2;
% options.x_axis = lags;
% options.error = 'c95';
% plot_areaerrorbar(acLS, options)
% box off;
% 

figure
title('Measure of Synchrony')
boxplot((1./[SIAwake' SILS']))
% Set the x-axis tick labels
xticklabels({'QW', 'LS'})
box off;
ylabel('Synchrony Index []')
xlim([0.5 2.5])

% Plot the shit

figure;
for iEpoch = 1:size(xlimits,1) %EPOCH 41 for Dataset 2 - Main Figure

	% Awake

	subplot(2,1,1)
	[~,idx1] = min(abs(binsAwake-xlimits(iEpoch,1)));
	[~,idx2] = min(abs(binsAwake-xlimits(iEpoch,2)));

	yyaxis left
	for iSpk = 1:length(spkTimesAwake)
		st = spkTimesAwake{iSpk}(spkTimesAwake{iSpk}>=xlimits(iEpoch,1) & spkTimesAwake{iSpk}<xlimits(iEpoch,2));
		if ~isempty(st)
			plot([st; st], [iSpk iSpk+0.5],'-k')
		end
		hold on
	end
	set(gca,'YTick',[])
	ylim([0, length(spkTimesAwake)+1])
	yyaxis right
	plot(binsAwake(idx1:idx2-1), (stateSeqAwake((idx1:idx2-1))),'LineWidth',0.75)
	set(gca,'XTick',[])
	set(gca,'YTick',[])
	box off;

	% LS

	subplot(2,1,2)
	[~,idx1] = min(abs(binsLS-xlimits(iEpoch,1)));
	[~,idx2] = min(abs(binsLS-xlimits(iEpoch,2)));

	yyaxis left
	for iSpk = 1:length(spkTimesLS)
		st = spkTimesLS{iSpk}(spkTimesLS{iSpk}>=xlimits(iEpoch,1) & spkTimesLS{iSpk}<xlimits(iEpoch,2));
		if ~isempty(st)
			plot([st; st], [iSpk iSpk+0.5],'-k')
		end
		hold on
	end
	ylim([0, length(spkTimesLS)+1])
	ylabel('Neuron ID')
	yyaxis right
	plot(binsLS(idx1:idx2-1), (stateSeqLS((idx1:idx2-1))),'LineWidth',0.75)
	ylabel('P(UP State)')
	set(gca,'XTick',[])
	box off;

	pause

	close all

end

%% Compute and plot the CWT power spectrum

% fs = params.Fs/2;
% if session == 1
% 	tMax = 1200;
% else
% 	tMax = 575;
% end
% sampMax = tMax*fs;
% 
% % Downsample again. We don't need 500Hz.
% lfpAwake = valLFPAwake(:,1:1:sampMax);
% tLFP = tLFPAwake(1:1:sampMax);
% lfpLS = valLFPLS(:,1:1:sampMax);
% lfpDS = valLFPDS(:,1:1:sampMax);
% 
% cwtAwake = zeros(size(lfpAwake,1),169,size(lfpAwake,2));
% cwtLS = zeros(size(lfpLS,1),169,size(lfpLS,2));
% cwtDS = zeros(size(lfpDS,1),169,size(lfpDS,2));
% 
% clear data; clear tmp;
% for iChan = 1:size(lfpAwake,1)
% 
% 	fprintf('Computing CWT for Chan : %d\n',iChan);
% 
% 	tic;
% 
% 	data = gpuArray(single(lfpAwake(iChan,:)));
% 	tmp = cwt(data,'morse',fs,'VoicesPerOctave',24,'NumOctaves',7,'WaveletParameters',[3 30]);
% 	tmp = double(gather(tmp));
% 	tmp = abs(tmp.^2);
% 	cwtAwake(iChan,:,:) = zscore(tmp,[],1);
% 	clear data; clear tmp;
% 
% 	data = gpuArray(single(lfpLS(iChan,:)));
% 	tmp = cwt(data,'morse',fs,'VoicesPerOctave',24,'NumOctaves',7,'WaveletParameters',[3 30]);
% 	tmp = double(gather(tmp));
% 	tmp = abs(tmp.^2);
% 	cwtLS(iChan,:,:) = zscore(tmp,[],1);
% 	clear data; clear tmp;
% 
% 	data = gpuArray(single(lfpDS(iChan,:)));
% 	[tmp,f] = cwt(data,'morse',fs,'VoicesPerOctave',24,'NumOctaves',7,'WaveletParameters',[3 30]);
% 	tmp = double(gather(tmp));
% 	tmp = abs(tmp.^2);
% 	cwtDS(iChan,:,:) = zscore(tmp,[],1);
% 	clear data; clear tmp;
% 
% 	toc;
% end
% 
% f = gather(f);
% 
% % Plot the shit
% 
% sgAwake = squeeze(nanmean(cwtAwake,1));
% sgLS = squeeze(nanmean(cwtLS,1));
% sgDS = squeeze(nanmean(cwtDS,1));
% 
% Yticks = 2.^(round(log2(min(f))):round(log2(max(f))));
% 
% if session == 1
% 
% 	xlimits(:,1) = 0:10:1190;
% 	xlimits(:,2) = 10:10:1200;
% 
% else
% 
% 	xlimits(:,1) = 0:10:560;
% 	xlimits(:,2) = 10:10:570;
% 
% end
% 
% iEpoch = 47;
% 
% [~,idx1] = min(abs(tLFP-xlimits(iEpoch,1)));
% [~,idx2] = min(abs(tLFP-xlimits(iEpoch,2)));
% 
% figure
% subplot(3,1,1)
% imagesc(tLFP(idx1:idx2),log2(f),sgAwake(:,idx1+1:idx2))
% AX = gca;
% set(AX, 'YTick',log2(Yticks(:)), 'YTickLabel',num2str(sprintf('%g\n',Yticks)))
% AX.YLim = log2([min(f), max(f)]);
% axis xy
% colormap jet
% AX.CLim = [-1 2.5];
% 
% subplot(3,1,2)
% imagesc(tLFP(idx1:idx2),log2(f),sgLS(:,idx1+1:idx2))
% AX = gca;
% set(AX, 'YTick',log2(Yticks(:)), 'YTickLabel',num2str(sprintf('%g\n',Yticks)))
% AX.YLim = log2([min(f), max(f)]);
% axis xy
% colormap jet
% AX.CLim = [-1 2.5];
% 
% subplot(3,1,3)
% imagesc(tLFP(idx1:idx2),log2(f),sgDS(:,idx1+1:idx2))
% AX = gca;
% set(AX, 'YTick',log2(Yticks(:)), 'YTickLabel',num2str(sprintf('%g\n',Yticks)))
% AX.YLim = log2([min(f), max(f)]);
% axis xy
% colormap jet
% AX.CLim = [-1 2.5];
% colorbar

%% Plot cycle frequencies

if session == 2
	cycDursAwake = dursUpStateAwake+dursDownStateAwake;
	cycDursLS = dursUpStateLS+dursDownStateLS;
	cycDursDS = dursUpStateDS+dursDownStateDS;
elseif session == 1
	cycDursAwake = dursUpStateAwake(1:end-1)+dursDownStateAwake;
	cycDursLS = dursUpStateLS(1:end-1)+dursDownStateLS;
	cycDursDS = dursUpStateDS+dursDownStateDS;

elseif session == 3

	cycDursAwake = dursUpStateAwake+dursDownStateAwake;
	cycDursLS = dursDownStateLS+dursUpStateLS(1:end-1);
	
end

subplot(1,2,1)
histogram(1./cycDursAwake,10,'Normalization','pdf','FaceColor','r','FaceAlpha',0.5)
box off;
vline(median(1./cycDursAwake),'--k')

subplot(1,2,2)
histogram(1./cycDursLS,10,'Normalization','pdf','FaceColor','g','FaceAlpha',0.5)
box off;
vline(median(1./cycDursLS),'--k')


%% Save all the figures

cd('C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\Results\PFC')

% % Create a sub-folder to hold the genWithFolds results separately. DO NOT PUT THEM IN THE MAIN DIRECTORY!
% mkdir genWithFolds
% cd genWithFolds
% mkdir ft0-125
% cd ft0-125
% mkdir(num2str(session))
% cd(num2str(session))
% if session == 3
%     mkdir(num2str(postFAHA))
%     cd(num2str(postFAHA))
% end
%
% % Get all open figure handles
% figure_handles = findall(0, 'Type', 'figure');
%
% % Loop through the figure handles and save each figure
% for iFig = 1:length(figure_handles)
%     fig = figure_handles(iFig);
%     % Set the desired file name and format for the saved figure
%     file_name = sprintf('Fig_%d.png', iFig);
%
%     % Save the figure
%     saveas(fig, file_name);
% end
% save('durationFrequencyStatistcs.mat','pVal','sig','-v7.3')
% pause(3)
% close all
