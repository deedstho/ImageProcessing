% Authors: Lauren Peterson, Thomas Deeds, Yiting Luo
% Last Updated: December 8, 2016


% Setup 

% vl_compilenn('enableGpu', true, ...
%   'enableImreadJpeg', true, ...
%   'cudaRoot', '/usr/um/cuda-7.5/', ...
%     'cudaMethod', 'nvcc');

 % load in values from input network
 in_net = load('imagenet-vgg-verydeep-19.mat');

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
des_res = run_net(input, in_net);

% Parameters for training
num_iterations = 100;
rate = .001;
loss_graph = [];
test_values = [];
for i = 1:num_iterations
    
    act_res = run_net(white_noise, in_net);


     % Calculate loss and partial derivative
    content_loss = loss(des_res.x5, act_res.x5);
    gradient = der_loss(des_res.x5, act_res.x5);

    % Save original values for graphing later
    loss_graph = vertcat(loss_graph, content_loss);

    % Run backwards through the network
    back_res = run_net(white_noise, in_net, act_res, gradient);
    
     % Change input image
    white_noise = white_noise - rate * back_res.dzdx0; 
    
     if (i == 1)
        figure;
        image(white_noise);
        title('1st Iteration');
    end
    
    if (i == num_iterations)
        figure;
        image(white_noise);
        title('Nth Iteration');
    end
end

%loss_graph = loss_graph(:,2:);
figure;
plot(loss_graph);

