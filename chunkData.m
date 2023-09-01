function out = chunkData(dataMatrix,Fs,chunkDuration)

% Assuming nChans x bBins is your data matrix named 'dataMatrix'
nChans = size(dataMatrix, 1);
bBins = size(dataMatrix, 2);

binsPerChunk = Fs * chunkDuration;

% Calculate the number of complete chunks
nChunks = floor(bBins / binsPerChunk);

% Truncate the data matrix to have only complete chunks
truncatedDataMatrix = dataMatrix(:, 1:(nChunks * binsPerChunk));

% Reshape the truncated data matrix into a 3D matrix with dimensions nChans x binsPerChunk x nChunks
out = reshape(truncatedDataMatrix, [nChans, binsPerChunk, nChunks]);
