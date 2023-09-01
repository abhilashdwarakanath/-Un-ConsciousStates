function [logLikelihood, AIC, BIC] = compute_goodness_of_fit(data, estTR, estEM, logLikelihood)

    % Calculate the number of parameters
    numStates = size(estTR, 1);
    numParameters = numStates * (numStates - 1) + numStates * (size(estEM, 2) - 1);

    % Compute the AIC and BIC values
    AIC = -2 * logLikelihood + 2 * numParameters;
    BIC = -2 * logLikelihood + numParameters * log(length(data));
end
