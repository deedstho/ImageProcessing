% PUT HEADER
function [output] = loss(desired_featmaps, actual_featmaps)

    % Compute loss
    output = 1/2 .* (actual_featmaps - desired_featmaps).^2;
    output = sum(sum(output));  
    output = reshape(output, [1,64]);

end