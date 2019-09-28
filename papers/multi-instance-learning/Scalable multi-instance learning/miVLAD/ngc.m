opt = InitSystem();
opt.kmeans_num_center = 2;
opt.PCA_energy = 0.0;
% The default parameters of miVLAD.

addpath('../data/figure');
addpath('../data/musk');
addpath('../data/ngc');
addpath('../data/mil');

inputname = 'ngc2.mat';
% You can choose different inputnames, e.g., "musk1", "musk2", "elephant", 
% "fox" and "tiger".
% The different inputnames indicates different data sets.

load(inputname);
if ndims(pos) == 3
    poscell = cell(1, size(pos,1));
    for ii = 1: size(pos,1)
        poscell{1, ii} = squeeze(pos(ii,:,:));
    end
    pos = poscell;
end
if ndims(neg) == 3
    negcell = cell(1, size(neg,1));
    for ii = 1: size(neg,1)
        negcell{1,ii} = squeeze(neg(ii,:,:));
    end
    neg = negcell;
end
pos_bag = size(pos,2);
neg_bag = size(neg,2);
num_bag = pos_bag + neg_bag;
%%
repeat = 100;
acc = zeros(repeat,1);
tp = zeros(repeat,1);
fp = zeros(repeat,1);
tn = zeros(repeat,1);
fn = zeros(repeat,1);
trainTime = zeros(repeat,1);
testTime = zeros(repeat,1);
auc = zeros(repeat,1);


for i = 1:repeat
    % cut data into postive bags and negative bags
    pos_perm = randperm(pos_bag);
    neg_perm = randperm(neg_bag);
    train_pos_ind = pos_perm(1:floor(pos_bag*0.7));
    test_pos_ind = pos_perm(floor(pos_bag*0.7)+1:end);
    train_neg_ind = neg_perm(1:floor(neg_bag*0.7));
    test_neg_ind = neg_perm(floor(neg_bag*0.7)+1:end);
    
    train_bag = [pos(train_pos_ind), neg(train_neg_ind)];
    train_label = [ones(size(train_pos_ind)), zeros(size(train_neg_ind))];
    test_bag = [pos(test_pos_ind), neg(test_neg_ind)];
    test_label = [ones(size(test_pos_ind)), zeros(size(test_neg_ind))];
    
    num_train_bag = size(train_bag, 2);
    num_test_bag = size(test_bag, 2);
    
    % Create codebook
    train_instances = [];
    for ii = 1: num_train_bag
        train_instances = [train_instances; train_bag{1,ii}];
    end
    [codes, opt] = CreateKmeansCodebook(train_instances,opt);
    
    % Convert data into VLAD format
    dim = opt.PCA_dim * opt.kmeans_num_center;
    vlad = zeros(num_bag, dim);
    for ii = 1:num_train_bag
        vlad(ii,:) = ExtractVLAD(train_bag{1,ii}, opt, codes);
    end
    for ii = num_train_bag + 1:num_bag
        vlad(ii,:) = ExtractVLAD(test_bag{1,ii - num_train_bag}, opt, codes);
    end
    
    minv = min(vlad);
    maxv = max(vlad) - minv;
    maxv = 1./maxv;
    vlad = (vlad -repmat(minv,num_bag,1)) .* repmat(maxv,num_bag,1);
    vlad(isnan(vlad))=0;

    vlad = sparse(vlad);
    % easyensemble index
    eesample = EasyEnsemble(size(train_pos_ind,2), size(train_neg_ind,2), 5);
    model = train(train_label(eesample)',vlad(eesample,:),'-s 2 -c 0.05 -B -1 -q');
    trainTime(i) = toc;
    tic
    [pred_label, accuracy, dec_val] = predict(test_label',vlad(1+num_train_bag:end,:),model);
    testTime(i) = toc;
    acc(i) = accuracy(1);
    tp(i) = sum((pred_label == 1) .* (test_label' == 1));
    fp(i) = sum((pred_label == 1) .* (test_label' == 0));
    tn(i) = sum((pred_label == 0) .* (test_label' == 0));
    fn(i) = sum((pred_label == 0) .* (test_label' == 1));
    [X, Y, T, AUCV] = perfcurve(test_label, dec_val, 1);
    auc(i) = AUCV;
end
disp(['The results of the ' inputname(1:(strfind(inputname,'.')-1)) ' data set are as follows:']);
disp(['Accuracy = ',num2str(mean(acc)),'¡À',num2str(std(acc(:)))]);
disp(['AUC = ', num2str(mean(auc))]); 
disp(['TP=',num2str(mean(tp)), '  FP', num2str(mean(fp))]);
disp(['TN=',num2str(mean(tn)), '  FN', num2str(mean(fn))]);