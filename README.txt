README.txt

This document describes how to run our content reconstruction. This is the 
simplest possible way to see what our project does, but it does require some
steps. Please don't hesitate to contact us if MatConvnet gives you trouble; 
we are happy to help.

SOURCES:

fox.jpg: http://shushi168.com/picture.html

vgg19_one_layer.mat: http://www.vlfeat.org/matconvnet/pretrained/\
(contains weights from "imagenet-vgg-verydeep-19.mat")

1. Downloading MatConvNet

Download or clone MatConvNet from https://github.com/vlfeat/matconvnet.git

2. Compiling MatConvNet

Run the following commands from the MATLAB command window:

	>> cd <MatConvNet Path>
	>> addpath matlab
	>> vl_compilenn

3. Setting up MatConvNet

Run the following command in matlab BEFORE EACH USE

	>> run <MatConvNet Path>/matlab/vl_setupnn

4. Running content_reconstruction.m

	>> content_reconstruction('fox.jpg', 0.01, 40)
