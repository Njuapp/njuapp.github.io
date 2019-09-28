function [feat] = ExtractVLAD(raw,opt,codes)

% create VLAD feature
num = size(raw,1);
if opt.PCA_energy>0
    raw = raw - repmat(codes.mu,num,1); % center the raw features
end
feat = VLAD(raw,codes.lf,codes.kmeans',opt);

function vladfeature = VLAD(feat,lf,codebook,opt)

if opt.PCA_energy>0
    temp = (feat*lf)';
else
    temp = feat';
end
[~, I] = min(vl_alldist2(codebook,temp),[],1);
vladfeature = zeros(size(codebook,2),size(lf,2));
appear = unique(I);
for i=1:length(appear)
    vladfeature(appear(i),:) = mean(temp(:,I==appear(i)),2)' - codebook(:,appear(i))';
end
vladfeature = vladfeature';
vladfeature = vladfeature(:)';
vladfeature = sign(vladfeature).*sqrt(vladfeature);