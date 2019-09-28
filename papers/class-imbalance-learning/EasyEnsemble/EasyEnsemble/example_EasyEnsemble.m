function example_EasyEnsemble

%--------------------------------------
% Dataset format instruction:
%   In sample data file "haberman.mat", training 'train' and test set 'test' are n-by-d matrixes,
%   where, d is the number of dimensions. And their labels
%   'traintarget' and 'testtarget' are n-by-1 vectors, with values of 0 (negative) and
%   1 (positive). 'catidx' indicates which attributes are discrete ones.  
%
% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
%--------------------------------------

load haberman.mat

T=4; % set parameter T (sample T subsets of negtive examples)
si=10; % set parameter si (use si iterations to train each AdaBoost classifier)

rand('state',sum(100*clock));
ensemble=EasyEnsemble(train,traintarget,catidx,T,si);
f = EvaluateValue(test,ensemble); % get real valued output
rates = CalculatePositives(ensemble,test,testtarget);
plot(rates(:,1),rates(:,2));
auc = CalculateAUC(rates)

re=f>=ensemble.thresh;
[fval,gmean] = ImbalanceEvaluate(testtarget,re,1,0)




