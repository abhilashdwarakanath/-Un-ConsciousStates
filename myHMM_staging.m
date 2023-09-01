clear all
clc
close all

%% Load data

cd('D:\RS_ICMS\J08\20210225\PFC\Sev1RSSt')
load neuralActivity.mat
spiking_raster = neuralActivity.spikes.activity;
t = neuralActivity.spikes.t;

%% Remove low-firing neurons and create the binned spike count structure

fr = sum(spiking_raster,2)./t(end);

[idx] = fr>0.5;

valid_spiking_raster = spiking_raster(idx,:);

clear spiking_raster;

tStart = t(1)*1e3;
tEnd = t(end)*1e3;

bins = tStart:50:tEnd; % bin spikes in 50ms bins

spikeCounts = zeros(size(valid_spiking_raster,1),length(bins));

for i = 1:size(valid_spiking_raster,1)

    spikes = find(valid_spiking_raster(i,:)==1);

    spikeTimes = t(spikes)*1e3;

    spikeCounts(i,:) = ((histc(spikeTimes,bins))./0.05).^2;

end

%% Fit the poisson HMM

x = 1;