% Authors: Lauren Peterson, Thomas Deeds, Yiting Luo
% Last Updated: December 8, 2016

run ~/Documents/MATLAB/matconvnet-1.0-beta23/matlab/vl_setupnn;

% Content Image
input = read_and_process('fox.jpg');
figure;
image(input);
title('Goal Image');

% White Noise Image
white_noise = create_white_noise();
figure;
image(white_noise);
title('White Noise Image');


for 



