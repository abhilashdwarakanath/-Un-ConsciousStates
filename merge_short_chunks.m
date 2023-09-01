function merged_sequence = merge_short_chunks(sequence, min_duration)
    len = numel(sequence);
    merged_sequence = sequence;
    idx = 1;

    while idx < len
        start_idx = idx;

        while idx < len && sequence(idx) == sequence(idx + 1)
            idx = idx + 1;
        end

        chunk_length = idx - start_idx + 1;

        if chunk_length < min_duration
            if start_idx > 1 && idx < len && sequence(start_idx - 1) == sequence(idx + 1)
                merged_sequence(start_idx:idx) = sequence(start_idx - 1);
            elseif start_idx > 1
                merged_sequence(start_idx:idx) = sequence(start_idx - 1);
            elseif idx < len
                merged_sequence(start_idx:idx) = sequence(idx + 1);
            end
        end

        idx = idx + 1;
    end
end