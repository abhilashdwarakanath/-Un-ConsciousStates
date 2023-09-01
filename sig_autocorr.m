function [acf, lags, synchronyIndex] = sig_autocorr(signal, Fs, maxLags)

% Compute autocorrelation
    N = length(signal);
    maxLagsSamples = round(maxLags * Fs);
    lags = (-maxLagsSamples:maxLagsSamples) / Fs;
    acf = zeros(1, 2*maxLagsSamples+1);
    
    for i = -maxLagsSamples:maxLagsSamples
        if i < 0
            temp = signal(1:end+i) .* signal(-i+1:end);
        elseif i > 0
            temp = signal(i+1:end) .* signal(1:end-i);
        else
            temp = signal .* signal;
        end
        acf(i+maxLagsSamples+1) = sum(temp) / (N - abs(i));
    end
    
    acf = (acf-mean(acf))./std(acf,1);

    positiveACF = acf(lags >= 0);
    synchronyIndex = sum(positiveACF);

    % Normalise by the number of observations
    numObservations = length(positiveACF);
    synchronyIndex = synchronyIndex / numObservations;

end
