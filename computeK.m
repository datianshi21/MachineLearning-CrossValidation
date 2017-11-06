function K = computeK(kernel_type, X, Z, param)
% function K = computeK(kernel_type, X, Z)
% computes a matrix K such that Kij=g(x,z);
% for three different function linear, rbf or polynomial.
%
% Input:
% kernel_type: either 'linear','polynomial','rbf'
% X: n input vectors of dimension d (dxn);
% Z: m input vectors of dimension d (dxm);
% param: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
%
% OUTPUT:
% K : nxm kernel matrix
%
%

if nargin<2,
	Z=X;
end;

%%YOUR CODE HERE
[d , n] = size(X);
[d , m] = size(Z);
if (strcmp(kernel_type , 'linear') == 1)
    K = X' * Z;
elseif (strcmp(kernel_type , 'rbf') == 1)
    K = exp(-param .* (repmat(sum(X .* X ,1)', 1 , m) - 2 .* X' * Z + repmat(sum(Z .* Z,1), n , 1)));
elseif (strcmp(kernel_type , 'poly') == 1)
    K = (X' * Z + 1) .^ param;
end;
