%**********************************************************
% Description: Parameters for hand written digit recognition using Neural Netwroks and Back Propagation algorithm. This file shall  be
% sourced wherever the parameters are used. Do not change parameters in this file from any other location apart form this file.
% Author: Gopalakrishna Hegde <gplhegde@gmail.com>
% Created: 30 Aug 2015
%**********************************************************

TRAIN_SET = 0.6;						   	   % 60% of total size
CROSS_VALIDATION_SET = 0.2;				   	   % 20% of total size
TEST_SET = 0.2;							       % 20% of total size
NO_HIDDEN_NODES = 25;
NO_OUTPUT_NODES = 10;						   % we have 10 digits
MAX_ITER = 50;
