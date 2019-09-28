function [feat] = ExtractFV(raw,opt,codes)

% create FV feature
num = size(raw,1);
if opt.PCA_energy>0
    raw = raw - repmat(codes.mu,num,1); % center the raw features
end
feat = Fisher(raw,codes,opt);

function fv = Fisher(feat,codes,opt)

if opt.PCA_energy>0
    temp = (feat*codes.lf)';
else
    temp = feat';
end
fv = vl_fisher(temp,codes.kmeans,codes.std,codes.priors,'improved')';