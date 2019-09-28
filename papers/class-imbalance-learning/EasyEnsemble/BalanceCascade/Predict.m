function result= Predict(dataset, ensemble)
% To predict labels for a set of examples using 
% AdaBoost/EasyEnsemble/BalanceCascade classifer
% Input:
%   dataset: n-by-d test set
%   ensemble: AdaBoost/EasyEnsemble/BalanceCascade classifer
% Output:
%   result: predicted labels of test examples

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

result = EvaluateValue(dataset,ensemble);
result = (result >= ensemble.thresh);
