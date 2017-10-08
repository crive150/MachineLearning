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

zeroGrad = zeros(size(theta));

h = sigmoid(X* theta);

% Theta 1 is not to be penalized in regularization so it uses original cost/gradient and stores it in these variables
[J, zeroGrad] = costFunction(theta, X, y);


%Cost Function
RJ= lambda /(2*m) * sum(theta(2:end).^2);
su = -y'*log(h) - (1-y)'*log(1-h);
J = (su / m) + RJ

% Gradient
S = h - y;
RG = lambda * theta / m;
grad = (S' * X)' / m + RG;

grad(1) = zeroGrad(1);


% =============================================================

end
