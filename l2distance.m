function D=l2distance(X,Z)
% function D=l2distance(X,Z)
%	
% Computes the Euclidean distance matrix. 
% Syntax:
% D=l2distance(X,Z)
% Input:
% X: dxn data matrix with n vectors (columns) of dimensionality d
% Z: dxm data matrix with m vectors (columns) of dimensionality d
%
% Output:
% Matrix D of size nxm 
% D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
%
% call with only one input:
% l2distance(X)=l2distance(X,X)
%

[d,n]=size(X);
% YOUR CODE (you can copy it from previous projects)
if (nargin==1)
    D = sqrt(abs(repmat(diag(X' * X) , 1 , n) - 2 .* X' * X + repmat(diag(X' * X)' , n , 1)));
else
    [d , m] = size(Z);
    D = sqrt(abs(repmat(diag(X' * X) , 1 , m) - 2 .* X' * Z + repmat(diag(Z' * Z)' , n , 1)));
end;

