% Authors: Lauren Peterson, Thomas Deeds, Yiting Luo
% Last Updated: December 8, 2016


% Setup 
cd matconvnet-1.0-beta23/matlab;

% vl_compilenn('enableGpu', true, ...
%   'enableImreadJpeg', true, ...
%   'cudaRoot', '/usr/um/cuda-7.5/', ...
%     'cudaMethod', 'nvcc');



vl_setupnn();


% Content Image
input = read_and_process('fox.jpg');
figure;
image(input);
title('Original Image');

% White Noise Image
white_noise = create_white_noise();
figure;
image(white_noise);
title('Input Image');

% Run content image through net
des_res = run_net(input);

% Parameters for training
num_iterations = 100;
rate = .01;
loss_graph = [];
test_values = [];
for i = 1:num_iterations
    
    act_res = run_net(white_noise);


     % Calculate loss and partial derivative
    content_loss = loss(desired_result.x2, actual_result.x2);
    gradient = der_loss(desired_result.x2, actual_result.x2);

    % Save original values for graphing later
    loss_graph = vertcat(loss_graph, content_loss);

    % Run backwards through the network
    back_result = big_net(white_noise, w, b, p, s, d, gradient);
    


