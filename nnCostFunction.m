function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,...
                                   num_labels, X, y, lambda)
% This computes the cost and gradient of single hidden layer neural network given the weights and the training set.
%
% Inputs:
%	nn_params: unrolled version of neural network weights corresponding to layer1 and layer2
%	input_layer_size: no of features
%	hidden_layer_size: no of nodes in the hidden layer
%	num_labels: no of nodes in the output layer
%	X: training set, each row must contain one example. No need to add the offset element.
%	y: output labels
%	lambda: regularization factor.
% Outputs:
%	J: cost of the overall network for given weights and training set.
%	grad: gradient of cost func wrt all weights.
% Usage:
%	[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%--------------------------------------------------------------------------------------------------------------------

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
VEC_IMPL = 1;	% want your code to run slowly? set this to 0 !

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1) X];	% add the offset element for each example
Del_1 = zeros(size(Theta1));
Del_2 = zeros(size(Theta2));

% Form output label vector. set the ith element to 1 if the label is i. rest to 0
if(VEC_IMPL == 1)
	validY = zeros(m, num_labels);
	for t=1:m
		validY(t,y(t)) = 1;
	end
end

% Non vectorized code. runs slow
if (VEC_IMPL == 0)
	for i = 1:m
		a1 = X(i,:)';
		z2 = Theta1*a1;
		a2 = [1; sigmoid(z2)];
		z3 = Theta2*a2;
		h = sigmoid(z3);
		digit = zeros(num_labels,1);
		digit(y(i)) = 1;
		% cost function
		J = J + sum((-digit.*log(h))-((1-digit).*log(1-h)));
		% gradients
		delta_3 = h-digit;
		delta_2 = (Theta2'*delta_3).*[1;sigmoidGradient(z2)];
		Del_2 = Del_2 + delta_3*a2';
		Del_1 = Del_1 + delta_2(2:end)*a1';
	end
else	% vectorized code. difficult to understand !
	z2 = X*Theta1';
	a2 = [ones(m,1) sigmoid(z2)];
	h = sigmoid(a2*Theta2');
	J = sum(sum((-validY.*log(h))-((1-validY).*log(1-h))));
	delta_3 = h-validY;
	delta_2 = (delta_3*Theta2).*[ones(m,1) sigmoidGradient(z2)];
	for t=1:m
		Del_2 = Del_2 + delta_3(t,:)'*a2(t,:);
		Del_1 = Del_1 + delta_2(t,2:end)'*X(t,:);
	end
end
J = J/m;
% add regularization term
J = J +(lambda/(2*m))* (sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));
Theta1_grad = Del_1/m;
Theta2_grad = Del_2/m;

% add regularization term
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
