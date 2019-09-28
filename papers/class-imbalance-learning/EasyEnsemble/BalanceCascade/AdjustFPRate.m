function thresh = AdjustFPRate(dataset, ensemble, fpr)
% To adjust threshold such that AdaBoost classifier's false positive rate is $fpr$
% Input:
%   dataset: n-by-d a set of negative examples
%   ensemble: AdaBoost classifier
%   fpr: the false positive rate to be achieved
% Output:
%   thresh: threshold of current AdaBoost classifier

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

result = EvaluateValue(dataset,ensemble);
result = sort(result);
thresh = result(round(size(dataset,1)*(1-fpr)));