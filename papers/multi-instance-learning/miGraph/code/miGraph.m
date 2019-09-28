function ret = miGraph(train_data,test_data,param)
%%  This function is the main function for calling migraph method
%   to use this function, LibSVM should be available.
%   refer: C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector
%   machines, Department of Computer Science and Information Engineering,
%   National Taiwan University, Taipei, Taiwan, Technical Report, 2001.
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Ms. Yu-Yin Sun (sunyy@lamda.nju.edu.cn). For any problem concerning the code,
%        please feel free to contact Ms. Sun.
%%  Some variables used in the code
%   
%       train_data: n*2 cell array. n is the number of bags. data{i,1} contains the instances in bag i and data{i,2} contains the label of bag i
%       test_data:  n*2 cell array. n is the number of bags. data{i,1} contains the instances in bag i and data{i,2} contains the label of bag i
%                               
%       param:  struct
%               param.c----CSVM parameter C
%               param.gamma----RBF kernel parameter $\gamma$
%               param.thr----The threshold used in computing the weight of each instance
%%  Reference:
%   Z.-H. Zhou, Y.-Y. Sun, and Y.-F. Li. Multi-instance learning by treating instances as non-i.i.d. samples. 
%   In: Proceedings of the 26th International Conference on Machine Learning (ICML'09), Montreal, Canada, 2009, pp.1249-1256.
%% End of Instruction



%%  Computing the weight of instances in each bags
weight_cell = computeWeight([train_data;test_data],param);

train_weight = weight_cell(1:size(train_data,1),:);%   train_weight contains the weights of each training bags
test_weight = weight_cell(1+size(train_data,1):end,:);%   test_weight contains the weights of each testing bags

clear weight_cell;
%%  End of computation.


%%  Training process: use LibSVM with a given kernel matrix
nargin_train = cell(5,1);
nargin_test = cell(5,1);

nargin_train{1,1} = train_data;
nargin_train{2,1} = train_data;
nargin_train{3,1} = train_weight;
nargin_train{4,1} = train_weight;
nargin_train{5,1} = param;

kernel_train = build_kernel(nargin_train);
kernel_train = [(1:size(kernel_train,1))',kernel_train];

train_label = cell2mat(train_data(:,2));

opt = ['-c ',num2str(param.c),' -t 4 -s 0'];

model = svmtrain(train_label,kernel_train,opt);
%%  End of Training process

%%  Testing process
nargin_test{1,1} = test_data;
nargin_test{2,1} = train_data;
nargin_test{3,1} = test_weight;
nargin_test{4,1} = train_weight;
nargin_test{5,1} = nargin_train{5,1};

kernel_test = build_kernel(nargin_test);
kernel_test = [(1:size(kernel_test,1))',kernel_test];

test_label = cell2mat(test_data(:,2));

[a,b,c] = svmpredict(test_label,kernel_test,model);

%%  End of Testing process


ret=b(1);%return the accuracy of prediction