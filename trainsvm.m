function [svmclassify,sv_i,alphas]=trainsvm(xTr,yTr, C,ktype,kpar);
% function [svmclassify,sv_i,alphas]=trainsvm(xTr,yTr, C,ktype,kpar);
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% C   : regularization constant (in front of loss)
% ktype : (linear, rbf, polynomial)
% 
% Output:
% svmclassify : a classifier (scmclassify(xTe) returns the predictions on xTe)
% sv_i : indices of support vectors
% alphas : a nx1 vector of alpha values
%
% Trains an SVM classifier with kernel (ktype) and parameters (C,kpar)
% on the data set (xTr,yTr)
%

%if nargin<5,kpar=1;end;
%yTr=yTr(:);
%svmclassify=@(xTe) (rand(1,size(xTe,2))>0.5).*2-1; %% classify everything randomly
%n=length(yTr);



disp('Generating Kernel ...')
% 
% YOUR CODE
%
K = computeK(ktype , xTr , xTr , kpar);

disp('Solving QP ...')
%
% YOUR CODE 
%
[H , q , Aeq , beq , lb , ub]=generateQP(K , yTr , C);
alphas = quadprog(H, q, [], [] , Aeq , beq , lb , ub);

disp('Recovering bias')
%
% YOUR CODE 
%
bias=recoverBias(K,yTr',alphas,C);

disp('Extracting support vectors ...')
%
% YOUR CODE 
%
sv_i = find(alphas > 10^(-5) & alphas < 9.999);

disp('Creating classifier ...')
%
% YOUR CODE 
%
svmclassify=@(xTe) (alphas' .* yTr * computeK(ktype , xTr , xTe , kpar))' + bias;

disp('Computing training error:') % this is optional, but interesting to see
%
% YOUR CODE 
%


