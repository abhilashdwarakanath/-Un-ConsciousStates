clear all
clc
close all

% Pooling results for PFC for durations and cycle frequency
% AD

%% Params

params.nSessions = 4;
params.nPhases = 3;

%% Loop through the datasets and pool the shit - EARLY

pooledCycleFreqsLSPre = cell(1,params.nPhases);
pooledCycleFreqsLSPost = cell(1,params.nPhases);

pooledDursUpStateLSPre = cell(1,params.nPhases);
pooledDursDownStateLSPre = cell(1,params.nPhases);

pooledDursUpStateLSPost = cell(1,params.nPhases);
pooledDursDownStateLSPost = cell(1,params.nPhases);

synchronyIndexLSPre = cell(1,params.nPhases);
synchronyIndexLSPost = cell(1,params.nPhases);

for iPhase = 1:params.nPhases

    for iSession = 1:params.nSessions

        if iSession == 3

            cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\Results\genWithFolds\Phases\ft0-05\' num2str(iSession) '\2\'])

        else

            cd(['C:\Users\Abhilash Dwarakanath\Documents\MATLAB\RS_ICMS\RS_Data\PFC\Results\genWithFolds\Phases\ft0-05\' num2str(iSession) '\'])

        end

            load durations.mat
            load synchronyIndex.mat

            upDursLSPre = state(2).durations(iPhase).UP; downDursLSPre = state(2).durations(iPhase).DOWN;
            upDursLSPost = state(3).durations(iPhase).UP; downDursLSPost = state(3).durations(iPhase).DOWN;

            nCyclesLSPre = min(length(upDursLSPre), length(downDursLSPre));
            cycleDursLSPre = 1./(upDursLSPre(1:nCyclesLSPre) + downDursLSPre(1:nCyclesLSPre));

            nCyclesLSPost = min(length(upDursLSPost), length(downDursLSPost));
            cycleDursLSPost = 1./(upDursLSPost(1:nCyclesLSPost) + downDursLSPost(1:nCyclesLSPost));

            pooledCycleFreqsLSPre{iPhase} = [pooledCycleFreqsLSPre{iPhase} cycleDursLSPre];
            pooledCycleFreqsLSPost{iPhase} = [pooledCycleFreqsLSPost{iPhase} cycleDursLSPost];

            pooledDursUpStateLSPre{iPhase} = [pooledDursUpStateLSPre{iPhase} upDursLSPre];
            pooledDursDownStateLSPre{iPhase} = [pooledDursDownStateLSPre{iPhase} downDursLSPre];

            pooledDursUpStateLSPost{iPhase} = [pooledDursUpStateLSPost{iPhase} upDursLSPost];
            pooledDursDownStateLSPost{iPhase} = [pooledDursDownStateLSPost{iPhase} downDursLSPost];

            synchronyIndexLSPre{iPhase} = [synchronyIndexLSPre{iPhase} SILSPre{iPhase}];
            synchronyIndexLSPost{iPhase} = [synchronyIndexLSPost{iPhase} SILSPost{iPhase}];

    end

end

%% Do pooled statistics

for iPhase = 1:params.nPhases

    [sigVal(iPhase).cycleFreqs] = ranksum(pooledCycleFreqsLSPre{iPhase},pooledCycleFreqsLSPost{iPhase});
    [sigVal(iPhase).upDurs] = ranksum(pooledDursUpStateLSPre{iPhase},pooledDursUpStateLSPost{iPhase});
    [sigVal(iPhase).downDurs] = ranksum(pooledDursDownStateLSPre{iPhase},pooledDursDownStateLSPost{iPhase});
    [~,sigVal(iPhase).si] = ttest(synchronyIndexLSPre{iPhase},synchronyIndexLSPost{iPhase});

end

%% Fit gamma distributions

freqBins = 0.001:0.25:8;
timeBins = 0:0.125:4;

for iPhase = 1:params.nPhases

    pars = gamfit((pooledCycleFreqsLSPre{iPhase}));
    gamFitCycleFreqsLSPre{iPhase} = gampdf(freqBins,pars(1),pars(2));

    pars = gamfit((pooledCycleFreqsLSPost{iPhase}));
    gamFitCycleFreqsLSPost{iPhase} = gampdf(freqBins,pars(1),pars(2));

    pars = gamfit(exp(pooledDursUpStateLSPre{iPhase}));
    gamFitUpDursLSPre{iPhase} = gampdf(timeBins,pars(1),pars(2));

    pars = gamfit(exp(pooledDursUpStateLSPre{iPhase}));
    gamFitUpDursLSPre{iPhase} = gampdf(timeBins,pars(1),pars(2));

    pars = gamfit(exp(pooledDursUpStateLSPost{iPhase}));
    gamFitUpDursLSPost{iPhase} = gampdf(timeBins,pars(1),pars(2));

    pars = gamfit(exp(pooledDursDownStateLSPre{iPhase}));
    gamFitDownDursLSPre{iPhase} = gampdf(timeBins,pars(1),pars(2));

    pars = gamfit(exp(pooledDursDownStateLSPost{iPhase}));
    gamFitDownDursLSPost{iPhase} = gampdf(timeBins,pars(1),pars(2));

end

%% Plot shit

phases = {'Early', 'Middle', 'Late'};

for iPhase = 1:params.nPhases

    figure;

    sgtitle(['RS Phase - ' phases{iPhase}])

    subplot(1,3,1)
    plot(freqBins,gamFitCycleFreqsLSPre{iPhase},'LineWidth',2)
    hold on
    plot(freqBins,gamFitCycleFreqsLSPost{iPhase},'LineWidth',2)
    box off
    xlabel('frequency [Hz]')
    ylabel('probability density []')
    title('UP/DOWN Cycle Frequency')


    subplot(1,3,2)
    semilogx(timeBins,gamFitUpDursLSPre{iPhase},'LineWidth',2)
    hold on
    semilogx(timeBins,gamFitUpDursLSPost{iPhase},'LineWidth',2)
    box off
    xlabel('duration [s]')
    title('UP state durations')


    subplot(1,3,3)
    semilogx(timeBins,gamFitDownDursLSPre{iPhase},'LineWidth',2)
    hold on
    semilogx(timeBins,gamFitDownDursLSPost{iPhase},'LineWidth',2)
    box off
    title('DOWN state durations')
    xlabel('duration [s]')
    legend('Pre Stimulation', 'Post Stimulation')

end

