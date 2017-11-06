function svmclassify=autosvm(xTr,yTr)
%	function svmclassify=autosvm(xTr,yTr)
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% 
% Output:
% svmclassify : a classifier (scmclassify(xTe) returns the predictions on xTe)
%
%
% Performs cross validation to train an SVM with optimal hyper-parameters on xTr,yTr
%%
disp('Performing cross validation ...');
%[bestC,bestP]=crossvalidate(xTr,yTr,'rbf',2.^[-1:8],2.^[-2:3]);
[bestC,bestP]=crossvalidate(xTr,yTr,'rbf',2.^[-1:3],2.^[1:3]);

newC = [(bestC-0.5):0.1:(bestC+0.5)];
newP = (bestP -0.5):0.1:(bestP+0.5);
[bestC,bestP]=crossvalidate(xTr,yTr,'rbf',newC,newP);
disp('Training SVM ...');
svmclassify=trainsvm(xTr,yTr,bestC,'rbf',bestP);
%testerr=sum(sign(svmclassify(xTe))~=yTe(:))/length(yTe)
