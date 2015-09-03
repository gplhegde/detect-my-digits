function g = sigmoidGradient(z)
% Computes gradient of the sigmoid function.
%
% Input:
%	z: input matrix
% Output:
%	g: gradient of the sigmoid(z)
% Usage:
%	g = sigmoidGradient(z)

g = zeros(size(z));
g = sigmoid(z).*(1-sigmoid(z));

end
