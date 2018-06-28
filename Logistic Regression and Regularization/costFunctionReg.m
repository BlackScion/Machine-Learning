function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sum1 = 0;
hyp = sigmoid(X * theta);
logist = (1/m) * (-y' * log(hyp) - (1 - y)' * log(1 - hyp));
for i = 2:size(theta, 1)
	sum1 = sum1 + theta(i)^2;
end
J = logist + (lambda/ (2*m)) * sum1;

theta_new = theta;
theta_new(1) = 0;

grad = (1/m) * X' * (hyp - y) + (lambda/m) * theta_new;




% =============================================================

end
