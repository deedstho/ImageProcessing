function [output_array] = read_and_process(input_jpeg)
%read_and_process reads and pre-processes a jpeg image

    % Convert jpeg to array
    output_array = imread(input_jpeg);
    
    % Resize image to 224 x 224 (source: https://arxiv.org/pdf/1409.1556.pdf)
    output_array = imcrop(output_array, [0 0 224 224]);

    mean_pixel_values(:, :, 1) = 124 * ones(224,224, 'uint8');
    mean_pixel_values(:, :, 2) = 117 * ones(224,224, 'uint8');
    mean_pixel_values(:, :, 3) = 104 * ones(224,224, 'uint8');
    output_array = output_array - mean_pixel_values;
    
    %figure;
    %image(output_array);
    %title('still uint8');
    
    % convert to single
    output_array = im2single(output_array);
    
    %figure;
    %image(output_array);
    %title('single'); 
       
end

