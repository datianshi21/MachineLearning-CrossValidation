function bias=recoverBias(K,yTr,alphas,C);
% function bias=recoverBias(K,yTr,alphas,C);
%
% INPUT:	
% K : nxn kernel matrix
% yTr : 1xn input labels
% alphas  : nx1 vector or alpha values
% C : regularization constant
% 
% Output:
% bias : the hyperplane bias of the kernel SVM specified by alphas
%
% Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
% 0<alpha<C
%


% YOUR CODE 
[d,n] = size(K);
%wx =alphas' .* yTr * diag(K);
CtoA = ones(n , 1) * C - alphas;
Ato0 = alphas - zeros(n , 1);
sub = abs(CtoA - Ato0);
[mininum , index] = min(sub);
wx = alphas' .* yTr' * K(:,index);
bias = 1/yTr(index) - wx;
