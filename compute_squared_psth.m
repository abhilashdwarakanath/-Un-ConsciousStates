function [psth_squared] = compute_squared_psth(spiking_raster, smoothing_kernel)
    % Check for input data dimensions
    if size(spiking_raster, 1) == 1 % 1xT data
        % Compute the PSTH using conv and square it
        psth = conv(spiking_raster, smoothing_kernel, 'same');
    else % NxT data
        % Compute the PSTH using conv2 and summing across the N channels
        psth = sum(conv2(spiking_raster, smoothing_kernel', 'same'), 1);
    end

    % Square the PSTH
    psth_squared = psth .^ 2;
end