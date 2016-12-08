run matconvnet-1.0-beta23/matlab/vl_setupnn;

input = read_and_process('fox.jpg');
figure;
image(input);
title('Original Image');

% Pull parameters from imagenet-vgg-verydeep-19.mat
    weights1 = in_net.layers{1,1}.weights{1,1};      % conv1_1
    biases1 = in_net.layers{1,1}.weights{1,2};
    pad1 = in_net.layers{1,1}.pad;
    stride1 = in_net.layers{1,1}.stride;
    dilate1 = in_net.layers{1,1}.dilate;
    
    weights3 = in_net.layers{1,3}.weights{1,1};       % conv1_2
    biases3 = in_net.layers{1,3}.weights{1,2};
    pad3 = in_net.layers{1,3}.pad;
    stride3 = in_net.layers{1,3}.stride;
    dilate3 = in_net.layers{1,3}.dilate;
    
    %pool5 = in_net.layers{1,5}.pool  ;           % pool1
    stride5 = in_net.layers{1,5}.stride;
    pad5 = in_net.layers{1,5}.pad;
    pool5 = in_net.layers{1,5}.pool;
    
    
white_noise = .1 * (randn(224,224,3, 'single'));
figure;
image(white_noise);
title('Input Image');

% Run the content image through the net
    des_res.x0 = input;
    des_res.x1 = vl_nnconv(des_res.x0, weights1, biases1,'pad', pad1, 'stride', stride1, ...
        'dilate', dilate1);
    des_res.x2 = vl_nnrelu(des_res.x1);
    des_res.x3 = vl_nnconv(des_res.x2, weights3, biases3,'pad', pad3, 'stride', stride3, ...
        'dilate', dilate3);
    des_res.x4 = vl_nnrelu(des_res.x3);
    des_res.x5 = vl_nnpool(des_res.x4, pool5, 'stride', stride5, 'pad', pad5);
    
% Iteration parameters   
num_iterations = 100;
rate = .001;
loss_graph = [];

for i = 1:num_iterations
    
    % Run forward through the network
    act_res.x0 = white_noise;
    act_res.x1 = vl_nnconv(act_res.x0, weights1, biases1,'pad', pad1, 'stride', stride1, ...
        'dilate', dilate1);
    act_res.x2 = vl_nnrelu(act_res.x1);
    act_res.x3 = vl_nnconv(act_res.x2, weights3, biases3,'pad', pad3, 'stride', stride3, ...
        'dilate', dilate3);
    act_res.x4 = vl_nnrelu(act_res.x3);
    act_res.x5 = vl_nnpool(act_res.x4, pool5, 'stride', stride5, 'pad', pad5);
    
    % Calculate loss and partial derivative
    content_loss = loss(des_res.x5, act_res.x5);
    gradient = der_loss(des_res.x5, act_res.x5);

    % Save original values for graphing later
    loss_graph = vertcat(loss_graph, content_loss);
    
    back_res.dzdx5 = gradient;
    back_res.dzdx4 = vl_nnpool(act_res.x4, pool5, back_res.dzdx5, 'stride', stride5, 'pad', pad5);
    back_res.dzdx3 = vl_nnrelu(act_res.x3, back_res.dzdx4);
    [back_res.dzdx2, back_res.dzdw3, back_res.dzdb3] = ...
    vl_nnconv(act_res.x2, weights3, biases3, back_res.dzdx3, 'pad', pad3, 'stride', stride3, 'dilate', dilate3) ;
    back_res.dzdx1 = vl_nnrelu(act_res.x1, back_res.dzdx2);
    [back_res.dzdx0, back_res.dzdw1, back_res.dzdb1] = ...
    vl_nnconv(act_res.x0, weights1, biases1, back_res.dzdx1, 'pad', pad1, 'stride', stride1, 'dilate', dilate1) ;
    
     % Update wieghts and biases
%     weights1 = weights1 - rate * back_res.dzdw1;
%     biases1 = biases1 - rate * back_res.dzdb1;
    
%     weights3 = weights3 - rate * back_res.dzdw3;
%     biases3 = biases3 - rate * back_res.dzdb3;
    
    
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
        title('100th Iteration');
    end
end 


%loss_graph = loss_graph(:,2:);
figure;
plot(loss_graph); % Still has the zeros in it
