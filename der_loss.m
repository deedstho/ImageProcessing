% PUT HEADER
function [output] = der_loss(desired_featmaps, actual_featmaps)
 % Derivative of the loss function
    output = actual_featmaps - desired_featmaps;
    
    %idx = find(actual_featmaps < 0);
    %output(idx) = 0;
    
    output(actual_featmaps < 0) = 0;

end