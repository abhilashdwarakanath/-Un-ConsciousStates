function [one_durations, zero_durations] = getBinaryChunkDurations(vector)

% This function accepts a binary vector and spits out the durations of the chunks of 1s and 0s.
% Use it for whatever, I don't care. This is adapted from the code used to estimate the duration distribution of chunks of Beta in Dwarakanath et al, 2023.


one_durations = [];
zero_durations = [];
curr_duration = 0;
curr_value = vector(1);

% Loop over the vector
for i = 1:length(vector)
    if vector(i) == curr_value
        curr_duration = curr_duration + 1;
    else
        if curr_value == 1
            one_durations = [one_durations, curr_duration];
        else
            zero_durations = [zero_durations, curr_duration];
        end
        curr_value = vector(i);
        curr_duration = 1;
    end
end

% Append the last duration to the appropriate list
if curr_value == 1
    one_durations = [one_durations, curr_duration];
else
    zero_durations = [zero_durations, curr_duration];
end