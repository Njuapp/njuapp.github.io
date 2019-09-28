function opt = InitSystem()

% Kmeans parameters
opt.kmeans_num_center = 2;

% PCA parameters
opt.PCA_dim = 0;
opt.PCA_energy = 0.0;

% setup VLFeat
run('/Users/hang/Documents/SMIL/vlfeat-0.9.21/toolbox/vl_setup');