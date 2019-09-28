function Accuracy = ten_fold_CV(data_file,index_path)
addpath('../code/');
load(data_file);
Accuracy=zeros(10,1);

param.c=100;
param.gamma=5;
param.thr=0.2;

for i=1:10
    load([index_path,'/index',num2str(i),'.mat']);
    train_data=data(trainIndex,:);
    test_data=data(testIndex,:);
    Accuracy(i,1)=miGraph(train_data,test_data,param);
end

