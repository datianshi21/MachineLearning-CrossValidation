function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
% function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
%
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% ktype : (linear, rbf, polynomial)
% Cs   : interval of regularization constant that should be tried out
% paras: interval of kernel parameters that should be tried out
% 
% Output:
% bestC: best performing constant C
% bestP: best performing kernel parameter
% bestval: best performing validation error
% allvalerrs: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)
%
% Trains an SVM classifier for all possible parameter settings in Cs and paras and identifies the best setting on a
% validation split. 
%

%%% Feel free to delete this
%bestC=0;
%bestP=0;
%bestval=10^10;

%% Split off validation data set
% YOUR CODE
% validationRate = 0.8;
% X = xTr;
% Y = yTr;
% [d,n]=size(X);
% itr=1:ceil(validationRate * n);
% ite=ceil(validationRate * n)+1:n;
% xTr=X(:,itr);
% yTr=Y(itr);
% xTv=X(:,ite);
% yTv=Y(ite);
k = 5;
[d , n] = size(xTr);
X = xTr;
Y = yTr;
bestval = 1;
iter1 = 1;
indices = crossvalind('Kfold', n , k);
indices = sort(indices);
for i = Cs
    iter2 = 1;
    for j = paras
        testerr = 0;
        for m = 1:k
            xTr = X(:,(indices ~= m));
            yTr = Y(indices ~= m);
            xTv = X(:,(indices == m));
            yTv = Y(indices == m);
            svmclassify = trainsvm(xTr,yTr,i,ktype,j);
            testerr = testerr + sum(sign(svmclassify(xTv))~=yTv(:))/length(yTv);
        end
        
        allvalerrs(iter1,iter2) = testerr / k;
        if (allvalerrs(iter1,iter2) < bestval)
            bestC = i;
            bestP = j;
            bestval = allvalerrs(iter1,iter2);
        end;
        iter2 = iter2  + 1;
    end;
    iter1 = iter1 + 1;
end;
%% Evaluate all parameter settings
% YOUR CODE
% bestval = 1;
% iter1 = 1;
% for (i = Cs)
%     iter2 = 1;
%     for(j = paras)
%         svmclassify = trainsvm(xTr,yTr,i,ktype,j);
%         allvalerrs(iter1,iter2) = sum(sign(svmclassify(xTv))~=yTv(:))/length(yTv);
%         if (allvalerrs(iter1,iter2) < bestval)
%             bestC = i;
%             bestP = j;
%             bestval = allvalerrs(iter1,iter2);
%         end;
%         iter2 = iter2  + 1;
%     end;
%     iter1 = iter1 + 1;
% end;
%% Identify best setting
% YOUR CODE


