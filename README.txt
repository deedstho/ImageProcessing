README.txt

This document describes how to run our program. This requires downloading, 
compiling, and seting up the MatConvNet Library. MatConvNet can optionally
be compiled with GPU support, but this is requires an NVIDIA GPU and we 
recommend only doing this locally using CAEN Linux. Knowledge of MATLAB is
highly recommended before attempting this process.

1. Downloading MatConvNet

Download or clone MatConvNet from https://github.com/vlfeat/matconvnet.git


2A. Compiling MatConvNet w/o GPU

Run the following commands from the MATLAB command window:

	>> cd <MatConvNet Path>
	>> addpath matlab
	>> vl_compilenn


--- OR ---


2B. Compiling MatConvNet w/ GPU

	>> cd <MatConvNet Path>
	>> addpath matlab
	>> vl_compilenn('enableGpu', true, ...
			'cudaRoot', '/usr/um/CUDA-7.5', ...
			'cudaMethod', 'nvcc')

3. Setting up MatConvNet

Run the following command in matlab BEFORE EACH USE

	>> run <MatConvNet Path>/matlab/vl_setupnn

4. Running FUNCTION_NAME
