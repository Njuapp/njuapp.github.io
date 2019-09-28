opt = InitSystem();
opt.kmeans_num_center = 2;
opt.PCA_energy = 0.0;
% The default parameters of miFV.

addpath('../data/figure');
addpath('../data/musk');

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
trainTime = zeros(num_fold,num_CV);
testTime = zeros(num_fold,num_CV);
for i = 1 : num_fold
    for j = 1 : num_CV
        tic
        cur_testIndex = testIndex((i-1)*num_CV+j,:);
        cur_trainIndex = 1:num_bag;
        cur_trainIndex(cur_testIndex) = [];
        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);

        % Create codebook
        train_instances = [];
        for ii = 1:num_train_bag
            train_instances = [train_instances; data{cur_trainIndex(ii),1}]; %#ok<AGROW>
        end
        [codes, opt] = CreateGMMCodebook(train_instances,opt);      

        % Convert data into FV format
        dim = opt.PCA_dim * opt.kmeans_num_center * 2;
        fv = zeros(num_bag,dim);
        labels = zeros(num_bag,1);
        for ii = 1:num_bag
            fv(ii,:) = ExtractFV(data{ii,1},opt,codes);
            fv(ii,:) = fv(ii,:) ./ norm(fv(ii,:));
            labels(ii) = data{ii,2};
        end

        minv = min(fv(cur_trainIndex,:));
        maxv = max(fv(cur_trainIndex,:)) - minv;
        maxv = 1./maxv;
        fv = (fv -repmat(minv,num_bag,1)) .* repmat(maxv,num_bag,1);
        fv(isnan(fv))=0;

        fv = sparse(fv);
        model = train(labels(cur_trainIndex),fv(cur_trainIndex,:),'-s 1 -c 0.05 -B -1 -q');
        trainTime(i,j) = toc;
        tic
        [pred_label, accuracy, dec_val] = predict(labels(cur_testIndex),fv(cur_testIndex,:),model);
        testTime(i,j) = toc;
        acc(i,j) = accuracy(1);
    end
end
acc = acc./100;
disp(' ');
disp(['The results of the ' inputname(1:(strfind(inputname,'.')-1)) ' data set are as follows:']);
disp(['Accuracy = ',num2str(mean(mean(acc))),'¡À',num2str(std(acc(:)))]);
disp(['TrainingTime = ',num2str(mean(mean(trainTime))),'¡À',num2str(std(trainTime(:)))]);
disp(['TestTime = ',num2str(mean(mean(testTime))),'¡À',num2str(std(testTime(:)))]);