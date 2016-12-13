function out = content_reconstruction( content_image_name, rate, num_iterations )

in_net = load('vgg19_one_layer.mat');

w = in_net.first_layer.weights{1,1};      % conv1_1
b = in_net.first_layer.weights{1,2};
p = in_net.first_layer.pad;
s = in_net.first_layer.stride;
d = in_net.first_layer.dilate;

content_image = read_and_process(content_image_name);

% Print the content image
figure;
image(content_image);
title('original image')

% Create a white noise image
white_noise = create_white_noise();

% Create feature maps of the content image
desired_result = one_layer_net(content_image, w, b, p, s, d);

% Converte rate to single
rate = single(rate);

% Used for saving loss values
loss_graph = [];

% Print the original white noise image
figure;
image(white_noise);
title('original white noise image');

for i = 1:num_iterations
    
    % Run forward
    actual_result = one_layer_net(white_noise, w, b, p, s, d);

     % Calculate loss and gradient of loss
    content_loss = loss(desired_result.x2, actual_result.x2);
    gradient = der_loss(desired_result.x2, actual_result.x2);

    % Save loss values
    loss_graph = vertcat(loss_graph, content_loss);

    % Run backwards through the network
    back_result = one_layer_net(white_noise, w, b, p, s, d, gradient);
    
    % Modify white noise image
    white_noise = white_noise - rate * back_result.dzdx1;
    
    % Plot the white noise image after 1 iteration
    if (i == 1)
        figure;
        image(white_noise);
        title('white noise after 1st iteration');
    end
    
    % Plot the final reconstructed image
    if (i == num_iterations)
        figure;
        image(white_noise);
        title('white noise after last iteration');
    end    
    
    %rate = rate / i;
end     


figure;
plot(loss_graph); % Still has the zeros in it

end


