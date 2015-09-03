function W = initWeights(inConnections, outConnections, epsilon)
% This will randomly initialize weight matrix of a neural network 
%
% Inputs:
%	inConnections: number of input connections to a neuron
%	outConnections: number of output connections to a neuron
%	epsilon: range of weights -epsilon to +epsilon
%
% Output:
%	W: weight matrix

W = zeros(outConnections, inConnections + 1);  % +1 is for bias term
W = rand(outConnections, inConnections + 1) * 2 * epsilon - epsilon;

end
