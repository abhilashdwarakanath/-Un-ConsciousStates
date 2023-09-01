function spikeTrains = simulateUPDOWNStatesGammaPoisson(params)

% This function requires these input arguments - 
%       
%           min and max firing rates for UP state
%           min and max firing rates for DOWN state
%           mean duration of UP and DOWN state
%           sampling rate
%           number of neurons to simulate
%           time to jitter in seconds
%
% AD. NS. 2023

%% Get stuff out

% Mean duration of each state (in seconds)
meanDownDur = params.meanDownDur;
meanUpDur = params.meanUpDur;

upFRRange = [params.upFR(1) params.upFR(2)];
downFRRange = [params.downFR(1) params.downFR(2)];

%% Generate UP/DOWN state durations with gamma-distributed durations

cycleDurs = [];
currIdx = 1;

while currIdx < params.duration * params.Fs

    % Draw down state duration from gamma distribution and convert to samples
    downDurSamps = round(gamrnd(0.75, meanDownDur) * params.Fs);

    % Draw up state duration from gamma distribution and convert to samples
    upDurSamps = round(gamrnd(0.9, meanUpDur) * params.Fs);

    % Add durations to the list
    cycleDurs = [cycleDurs; downDurSamps, upDurSamps]; %#ok<*AGROW>
    currIdx = currIdx + downDurSamps + upDurSamps;

end

%% Generate spike trains for each neuron based on common UP/DOWN states with different firing rates

spikeTrains = zeros(params.nNeurons, params.duration*params.Fs);
maxIdx = (params.duration*params.Fs)+1;

for iChan=1:params.nNeurons

    % Randomly set firing rates for each neuron in the up and down states from the range specified
    rateUp = upFRRange(1)+(upFRRange(2)-upFRRange(1))*rand;
    rateDown = downFRRange(1)+(downFRRange(2) - downFRRange(1))*rand;

    currIdx = 1;

    for iCycle=1:size(cycleDurs, 1)

        downDurSamps = cycleDurs(iCycle, 1);
        upDurSamps = cycleDurs(iCycle, 2);

        % Generate spikes during down state (random spikes) drawn from poisson distribution
        % Add a jitter to the start of each phase
        downIdxs = rand(1, downDurSamps) <= (rateDown / params.Fs);
        chunkIdxs = currIdx:currIdx + downDurSamps - 1;
        jitter = round((-params.jitter+(params.jitter--params.jitter)*rand)*params.Fs);
        chunkIdxs = chunkIdxs + jitter;
        chunkIdxs(chunkIdxs < 1) = 1;
        chunkIdxs(chunkIdxs > maxIdx) = maxIdx;
        spikeTrains(iChan, chunkIdxs) = downIdxs;
        
        currIdx = currIdx + downDurSamps;

        % Generate spikes during up state (coordinated spikes) drawn from poisson distribution
        % Add a jitter to the start of each phase
        upIdxs = rand(1, upDurSamps) <= (rateUp / params.Fs);
        chunkIdxs = currIdx:currIdx + upDurSamps - 1;
        jitter = round((-params.jitter+(params.jitter--params.jitter)*rand)*params.Fs);
        chunkIdxs = chunkIdxs + jitter;
        chunkIdxs(chunkIdxs < 1) = 1;
        chunkIdxs(chunkIdxs > maxIdx) = maxIdx;
        spikeTrains(iChan, chunkIdxs) = upIdxs;

        %initialise the next chunk
        currIdx = currIdx + upDurSamps;

    end
end
