function [ white_noise_out ] = create_white_noise(  )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Create white noise
    white_noise_out =  ones(224,224,3, 'single');
    
    %white_noise_out = read_and_process('star.jpg');
    
    % Create mean pixel values
    %mean_pixel_values(:, :, 1) = 124 * ones(224,224, 'single');
    %mean_pixel_values(:, :, 2) = 117 * ones(224,224, 'single');
    %mean_pixel_values(:, :, 3) = 104 * ones(224,224, 'single');
    
    % Subtract mean pixel values from white noise
    %white_noise_out = white_noise_out - mean_pixel_values;   
   
   
    % Print
    %figure
    %image(white_noise_out)

end

