function ensemble= EasyEnsemble(trainset, traintarget, catidx, T, rounds)

% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target
%   catidx: indicates which attributes are discrete ones (Note: start from 1)
%   T: sample $T$ subsets of negtive examples
%   rounds: use $rounds$ iterations to train each AdaBoost classifier
% Output:
%   ensemble: EasyEnsemble classifier, a structure variable

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)


poscount = sum(traintarget==1);
negcount = length(traintarget)-poscount;
posset = trainset(traintarget==1,:);
negset = trainset(traintarget==0,:);
negset = negset(randperm(negcount),:);

ensemble = struct('trees',{},'alpha',{},'thresh',{});

for node=1:T % stopping criteria
    nset = negset(1:poscount,:); % a ramdom subset of negtive examples
    curtrainset = [posset;nset];
    curtarget = zeros(size(curtrainset,1),1);
    curtarget(1:poscount)=1;
    ens = AdaBoost(curtrainset,curtarget,catidx,rounds);% node classifier    
    ensemble(node) = ens;    
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