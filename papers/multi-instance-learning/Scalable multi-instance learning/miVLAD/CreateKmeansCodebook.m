function [codes opt] = CreateKmeansCodebook(feats,opt)

tic;
codes = [];

if opt.PCA_energy>0
    codes.mu = mean(feats);
    feats = feats - repmat(codes.mu,size(feats,1),1); % center all features

    [temp , ~, latent] = princomp(feats);
    score = cumsum(latent)/sum(latent);
    opt.PCA_dim = min(find(score>opt.PCA_energy)); %#ok<MXFND>
    codes.lf = temp(:,1:opt.PCA_dim); % PCA load factors
    fprintf('    PCA features reserved energy: %.2f%% with %d features\n',score(opt.PCA_dim)*100,opt.PCA_dim);

    temp = (feats*codes.lf)'; % do PCA to the features
    [kmc, ~] = vl_kmeans(temp,opt.kmeans_num_center); % build a codebook
else
    opt.PCA_dim = size(feats,2);
    codes.lf = zeros(size(feats,2),size(feats,2));
    [kmc, ~] = vl_kmeans(feats',opt.kmeans_num_center); % build a codebook without PCA
end

codes.kmeans = kmc';
fprintf('    Features clustered into codewords.\n');

elapsed = toc;
fprintf('    Kmeans codebook generated in %f seconds.\n',elapsed);
