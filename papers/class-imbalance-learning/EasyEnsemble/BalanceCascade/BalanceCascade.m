function ensemble = BalanceCascade(trainset, traintarget, catidx, T, rounds)

% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target
%   catidx: indicates which attributes are discrete ones (Note: start from 1)
%   T: sample $T$ subsets of negtive examples
%   rounds: use $rounds$ iterations to train each AdaBoost classifier
% Output:
%   ensemble: BalanceCascade classifier, a structure variable

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)


poscount = sum(traintarget==1);
negcount = length(traintarget)-poscount;

posset = trainset(traintarget==1,:);
negset = trainset(traintarget==0,:);
negset = negset(randperm(negcount),:);

FP=(poscount/negcount)^(1/(T-1));

ensemble = struct('trees',{},'alpha',{},'thresh',{});


for node=1:T 
    nset = negset(1:poscount,:);
    curtrainset = [posset;nset];
    curtarget = zeros(size(curtrainset,1),1);
    curtarget(1:poscount)=1;
    ens = AdaBoost(curtrainset,curtarget,catidx,rounds);% node classifier
    ens.thresh = AdjustFPRate(negset, ens, FP);% base on negset
    ensemble(node) = ens;
    
    result = Predict(negset,ens);
    negset = negset(result==1,:); % remove correctly classified negtive examples
    negcount = size(negset,1);
    negset = negset(randperm(negcount),:);
end

%combine all weak learners to form the final ensemble
depth = length(ensemble);
ens= struct('trees',{},'alpha',{},'thresh',{});
for i=1:depth
   ens(1).trees = [ens.trees; ensemble(i).trees];
   ens(1).alpha = [ens.alpha; ensemble(i).alpha];
end
ens.thresh = sum(ens.alpha)/2;
ensemble = ens;