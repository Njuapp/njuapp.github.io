opt = InitSystem();
opt.kmeans_num_center = 2;
opt.PCA_energy = 0.0;
% The default parameters of miVLAD.

addpath('../data/figure');
addpath('../data/musk');
addpath('../data/ngc');

inputname = 'musk1.mat';
% You can choose different inputnames, e.g., "musk1", "musk2", "elephant", 
% "fox" and "tiger".
% The different inputnames indicates different data sets.

load(inputname);
num_bag = size(data,1);

str = 'musk1_Index.mat'; 
% The different ".mat" files indicates different corresponding index files.
% "musk1_Index" is for the musk1 data set. "musk2_Index" is for the musk2 
% data set. "figure_testIndex" is for the other image benchmark data sets, 
% i.e., Elephant, Fox, and Tiger.

load(str);
%%
num_fold = 10;
num_CV = 10;
acc = zeros(num_fold,num_CV);
tp = zeros(num_fold,num_CV);
fp = zeros(num_fold,num_CV);
tn = zeros(num_fold,num_CV);
fn = zeros(num_fold,num_CV);
trainTime = zeros(num_fold,num_CV);
testTime = zeros(num_fold,num_CV);
auctable = zeros(num_fold,num_CV);

for i = 1:num_fold
    for j = 1:num_CV
        tic
        cur_testIndex = testIndex((i-1)*num_CV+j,:);
        cur_trainIndex = 1:num_bag;
        cur_trainIndex(cur_testIndex) = [];
        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);

        % Create codebook
        train_instances = [];
        for ii = 1:num_train_bag
            train_instances = [train_instances; data{cur_trainIndex(ii),1}];
        end
        [codes, opt] = CreateKmeansCodebook(train_instances,opt);
        
        % Convert data into VLAD format
        dim = opt.PCA_dim * opt.kmeans_num_center;
        vlad = zeros(num_bag,dim);
        labels = zeros(num_bag,1);
        for ii = 1:num_bag
            vlad(ii,:) = ExtractVLAD(data{ii,1},opt,codes);
            labels(ii) = data{ii,2};
        end
        
        minv = min(vlad(cur_trainIndex,:));
        maxv = max(vlad(cur_trainIndex,:)) - minv;
        maxv = 1./maxv;
        vlad = (vlad -repmat(minv,num_bag,1)) .* repmat(maxv,num_bag,1);
        vlad(isnan(vlad))=0;
        
        vlad = sparse(vlad);
        model = train(labels(cur_trainIndex),vlad(cur_trainIndex,:),'-s 2 -c 0.05 -B -1 -q');
        trainTime(i,j) = toc;
        tic
        [pred_label, accuracy, dec_val] = predict(labels(cur_testIndex),vlad(cur_testIndex,:),model);
        testTime(i,j) = toc;
        acc(i,j) = accuracy(1);
        tp(i,j) = sum((pred_label == 1) .* (labels(cur_testIndex) == 1));
        fp(i,j) = sum((pred_label == 1) .* (labels(cur_testIndex) == 0));
        tn(i,j) = sum((pred_label == 0) .* (labels(cur_testIndex) == 0));
        fn(i,j) = sum((pred_label == 0) .* (labels(cur_testIndex) == 1));
        [X, Y, T, AUC] = perfcurve(labels(cur_testIndex), dec_val, 1);
        auctable(i,j) = AUC;
    end
end
acc = acc./100;
disp(' ');
disp(['The results of the ' inputname(1:(strfind(inputname,'.')-1)) ' data set are as follows:']);
disp(['Accuracy = ',num2str(mean(mean(acc))),'¡À',num2str(std(acc(:)))]);
disp(['TrainingTime = ',num2str(mean(mean(trainTime))),'¡À',num2str(std(trainTime(:)))]);
disp(['TestTime = ',num2str(mean(mean(testTime))),'¡À',num2str(std(testTime(:)))]);
disp(['AUC = ', num2str(mean(mean(auctable)))]); 
disp(['TP=',num2str(mean(mean(tp))), 'FP', num2str(mean(mean(fp)))]);
disp(['TN=',num2str(mean(mean(tn))), 'FN', num2str(mean(mean(fn)))]);