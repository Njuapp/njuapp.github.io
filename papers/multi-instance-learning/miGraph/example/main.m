function main()
data_path='data/alt.atheism.mat';

load(data_path);

%%  Get 10 fold CV partition for 10 times
positive=find(cell2mat(data(:,2))==1);
negative=find(cell2mat(data(:,2))~=1);

clear data;

get_ten_fold_CV(positive,negative);
%% End
for i=1:10
    index_path=['index/',num2str(i)];
    %%  Run 10 CV for 10 times
    acc(:,i)=ten_fold_CV(data_path,index_path);
    disp(acc(:, i));
    %%  End
end
disp('-------------------Accuracy-------------------');
disp(acc);