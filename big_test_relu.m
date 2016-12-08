run matconvnet-1.0-beta23/matlab/vl_setupnn;

in_net = load('imagenet-vgg-verydeep-19.mat');

w = in_net.layers{1,1}.weights{1,1};      % conv1_1
b = in_net.layers{1,1}.weights{1,2};
p = in_net.layers{1,1}.pad;
s = in_net.layers{1,1}.stride;
d = in_net.layers{1,1}.dilate;



input = read_and_process('fox.jpg');

figure;
image(input);
title('original image')

white_noise = create_white_noise();

desired_result = big_net_relu(input, w, b, p, s, d);

% Parameters for training
num_iterations = 10;
rate = single(.2);
loss_graph = [];
test_values = [];

%shrinkRate = 0.0001;
%momentum = 0.9;

%w_momentum = zeros('like', w) ;
%b_momentum = zeros('like', b) ;

for i = 1:num_iterations
    actual_result = big_net_relu(white_noise, w, b, p, s, d);

     % Calculate loss and partial derivative
    content_loss = loss(desired_result.x3, actual_result.x3);
    gradient = der_loss(desired_result.x3, actual_result.x3);

    % Save original values for graphing later
    loss_graph = vertcat(loss_graph, content_loss(1));

    % Run backwards through the network
    back_result = big_net_relu(white_noise, w, b, p, s, d, gradient);
    
    %w_momentum = momentum * w_momentum + rate * (back_result.dzdw + shrinkRate * w) ;
    %b_momentum = momentum * b_momentum + rate * 0.1 * back_result.dzdb ;
    
    %w = w - w_momentum;
    %b = b - b_momentum;
    
    % Update wieghts and biases
    w = w - rate * back_result.dzdw;
    b = b - rate * back_result.dzdb;
    
    % Change input image
    white_noise = white_noise - rate * back_result.dzdx1;
    
    test_values = vertcat(test_values, back_result.dzdx1);
    
    if (i == 1)
        figure;
        image(white_noise);
        title('1st iteration');
    end
    
    if (i == 10)
        figure;
        image(white_noise);
        title('10th iteration');
    end    
    
    %rate = rate / i;
end     

%loss_graph = loss_graph(:,2:);
%figure;
%plot(loss_graph); % Still has the zeros in it

%figure;
%image(white_noise);
%title('white noise');

%figure;
%image(actual_result.x2(:,:,1));
%title('conv result');




