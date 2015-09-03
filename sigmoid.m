function s = sigmoid(z)
% Sigmoid function
%
% Inputs:
%	z: input vector
% Output:
%	s: sigmoid of input vector
% Usage:
%	s = sigmoid(z);

s = 1.0 ./ (1.0 + exp(-z));
end
