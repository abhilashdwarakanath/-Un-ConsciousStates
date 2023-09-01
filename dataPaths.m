cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\11'])
load neuralActivity.mat

rasterDataAwake = neuralActivity.spikes.activity;
tAwake = neuralActivity.spikes.t;

cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\21'])
load neuralActivity.mat

rasterDataLSPre = neuralActivity.spikes.activity;
tLSPre = neuralActivity.spikes.t;

cd(['C:\Users\AD263755\Documents\MATLAB\UpDownStatesCodes\myHmmAD\PFC\' num2str(session) '\' '2' num2str(postFAHA)])
load neuralActivity.mat

rasterDataLSPost = neuralActivity.spikes.activity;
tLSPost = neuralActivity.spikes.t;
