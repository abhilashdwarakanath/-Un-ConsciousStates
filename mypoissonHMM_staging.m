clear all
clc
close all

%% Load data

cd('D:\RS_ICMS\J08\20210225\PFC\Sev1RSSt')
load neuralActivity.mat
spiking_raster = neuralActivity.spikes.activity;
t = neuralActivity.spikes.t;

%% Remove low-firing neurons and create raster

fr = sum(spiking_raster,2)./t(end);

[idx] = fr>0.5;
valid_spiking_raster = spiking_raster(idx,:);
