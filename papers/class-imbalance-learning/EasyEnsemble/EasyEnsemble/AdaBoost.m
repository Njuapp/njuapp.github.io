function ensemble = AdaBoost(trainset, traintarget, catidx, rounds)

% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target
%   catidx: indicates which attributes are discrete ones (Note: start from 1)
%   rounds: use $rounds$ iterations to train each AdaBoost classifier
% Output:
%   ensemble: AdaBoost classifier, a structure variable
% Note: this method uses 'treefit' decision tree method in statistics toolbox to 
%   generate base classifiers 

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

ensemble.trees = cell(rounds,1);
ensemble.alpha = zeros(rounds,1);
ensemble.thresh = 0;

weight = zeros(size(traintarget));
weight(traintarget==1) = 1/sum(traintarget==1);
weight(traintarget==0) = 1/sum(traintarget==0);
weight = weight / sum(weight);

result = zeros(size(traintarget));
for i=1:rounds
    [boostset boosttarget] = boost_data(trainset,traintarget,weight);
    tree = treefit(boostset,boosttarget,'method','classification','categorical',catidx);%base weak learner
    ensemble.trees{i} = tree;
    trainresult = treeval(tree,trainset)-1;
    trainerror = sum(weight.*(trainresult~=traintarget));
    beta = (1-trainerror)/trainerror;
    ensemble.alpha(i) = 0.5*log(beta);
    result = result + ensemble.alpha(i) * trainresult;
    weight = weight .* exp(-ensemble.alpha(i)*(trainresult-0.5).*(traintarget-0.5)*4);
    weight = weight / sum(weight);
end
ensemble.thresh = sum(ensemble.alpha)/2;