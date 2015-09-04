%**********************************************************
% Description: Hand written digit recognition using Neural network approach.
% Author: Gopalakrishna Hegde <gplhegde@gmail.com>
% Created: 30 Aug 2015
%**********************************************************

clear all;
close all;
clc;

% Load the image dataset. The images are stored in rows of matrix X. 
%Each image is reshaped into a column vector.
fprintf('Loading image dataset......\n');
load('dataset.mat');

% Source the config parameters for neural net.
source config.m;

% Data set size 
TOTAL_IMAGES = size(X, 1);
NO_FEATURES = size(X, 2);

% Shuffle the rows randomly
order = randperm(TOTAL_IMAGES);
X = X(order, :);
y = y(order, :);

% Set parameters for test set, cross validation set and training set
TRAIN_IMAGES = floor(TRAIN_SET * TOTAL_IMAGES);
CROSS_VALIDATION_IMAGES = floor(CROSS_VALIDATION_SET * TOTAL_IMAGES);
TEST_IMAGES = floor(TEST_SET * TOTAL_IMAGES);

fprintf('\n---------Parameters-------------\n');
fprintf(['Dataset size = %d\nTotal Features = %d\nTraining Set Size = %d\n',...
	'Cross Validation Set Size = %d\nTest Set Size = %d\n',...
	'No of Hidden nodes = %d\nNo of output nodes = %d\nMax iterations for cost function minimization = %d\n'],...
	TOTAL_IMAGES, NO_FEATURES, TRAIN_IMAGES, CROSS_VALIDATION_IMAGES, TEST_IMAGES, ...
	NO_HIDDEN_NODES, NO_OUTPUT_NODES, MAX_ITER);
fprintf('\n--------------------------------\n');

% Randomly initialize the neural network weights
epsilon = 0.12;
l1L2Weights = initWeights(NO_FEATURES, NO_HIDDEN_NODES, epsilon);
l2L3Weights = initWeights(NO_HIDDEN_NODES, NO_OUTPUT_NODES, epsilon);

% unroll weights
initialWeights = [l1L2Weights(:) ; l2L3Weights(:)];

% Setup function handler for cost and gradient function
options = optimset('MaxIter', MAX_ITER);
lambda = 0.8;
costFunction = @(p) nnCostFunction(p, NO_FEATURES, ...
                                   NO_HIDDEN_NODES,...
                                   NO_OUTPUT_NODES, X(1:TRAIN_IMAGES,:), ...
				   y(1:TRAIN_IMAGES,:), lambda);

% Minimize the cost w.r.t the weights using function minimization utility
[finalWeights, cost] = fmincg(costFunction, initialWeights, options);


% Reshape the final weights
l1L2Weights = reshape(finalWeights(1: NO_HIDDEN_NODES * (NO_FEATURES + 1)), ...
                 NO_HIDDEN_NODES, (NO_FEATURES + 1));

l2L3Weights = reshape(finalWeights((1 + (NO_HIDDEN_NODES * (NO_FEATURES + 1))):end), ...
                  NO_OUTPUT_NODES, (NO_HIDDEN_NODES + 1));
fprintf('Neural network training is completed......\n');
% Find the training accuracy
trainPred = predict(l1L2Weights, l2L3Weights, X(1:TRAIN_IMAGES,:));
fprintf('\nTraining Accuracy: %f\n', mean(double(trainPred == y(1:TRAIN_IMAGES,:))) * 100);

% Find testing accuracy
testPred = predict(l1L2Weights, l2L3Weights, ...
		X((TRAIN_IMAGES + CROSS_VALIDATION_IMAGES + 1):TOTAL_IMAGES,:));
fprintf('\nTesting Accuracy: %f\n',...
	 mean(double(testPred == y((TRAIN_IMAGES + CROSS_VALIDATION_IMAGES + 1):TOTAL_IMAGES,:))) * 100);

fprintf('\n------------END-------------\n');
