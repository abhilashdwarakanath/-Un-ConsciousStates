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