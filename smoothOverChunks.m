function modified_vec = smoothOverChunks(vec,min_duration,Fs)

    n = length(vec);
    modified_vec = vec;
    w = min_duration*Fs;
    
    % Iterate through the vector
    for i = 1:n
        count = 1;
        current_val = modified_vec(i);
        
        % Check the length of the current sequence
        while i+count <= n && modified_vec(i+count) == current_val
            count = count + 1;
        end
        
        % If the sequence is shorter than w, replace it with the flanking values
        if count < w
            left_flank = 0;
            right_flank = 0;
            
            if i > 1
                left_flank = modified_vec(i-1);
            end
            
            if i+count <= n
                right_flank = modified_vec(i+count);
            end
            
            if left_flank == right_flank
                modified_vec(i:i+count-1) = left_flank;
            end
        end
        
        % Move the iterator to the end of the current sequence
        i = i + count - 1;
    end
end