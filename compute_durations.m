function chunk_durations = compute_durations(sequence, min_duration)
    if nargin < 2
        min_duration = 10; % 0.1 seconds * 100 Hz
    end

    merged_sequence = merge_short_chunks(sequence, min_duration);
    chunk_durations = get_durations(merged_sequence);
end

function merged_sequence = merge_short_chunks(sequence, min_duration)
    chunk_starts = [1, find(diff(sequence)) + 1];
    chunk_lengths = diff([chunk_starts, numel(sequence) + 1]);

    merged_sequence = sequence;

    for i = 1:numel(chunk_lengths)
        if chunk_lengths(i) < min_duration
            if i > 1 && i < numel(chunk_lengths)
                if chunk_lengths(i - 1) > chunk_lengths(i + 1)
                    merged_sequence(chunk_starts(i):chunk_starts(i) + chunk_lengths(i) - 1) = sequence(chunk_starts(i) - 1);
                else
                    merged_sequence(chunk_starts(i):chunk_starts(i) + chunk_lengths(i) - 1) = sequence(chunk_starts(i) + chunk_lengths(i));
                end
            elseif i > 1
                merged_sequence(chunk_starts(i):chunk_starts(i) + chunk_lengths(i) - 1) = sequence(chunk_starts(i) - 1);
            elseif i < numel(chunk_lengths)
                merged_sequence(chunk_starts(i):chunk_starts(i) + chunk_lengths(i) - 1) = sequence(chunk_starts(i) + chunk_lengths(i));
            end
        end
    end
end

function durations = get_durations(sequence)
    len = numel(sequence);
    durations = [];
    count = 1;

    for i = 2:len
        if sequence(i) == sequence(i - 1)
            count = count + 1;
        else
            durations = [durations; sequence(i - 1), count];
            count = 1;
        end
    end

    durations = [durations; sequence(end), count];
end