function k = build_kernel(nargin)
%%  This function is used for computing kernel
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
%%  Some varables used in the code
%   
%       nargin{1,1} is the first data: n*2 cell array, the first column contains the data and the second contains their labels
%       nargin{2,1} is the second data: m*2 cell array, the first column contains the data and the second contains their labels
%       nargin{3,1} is the weight of all the bags in data1: n*1 cell array, each cell contains a n_i*n_i matrix
%       nargin{4,1} is the weight of all the bags in data2: m*1 cell array, each cell contains a n_i*n_i matrix
%       nargin{5,1} is the parameter used for computing kernel
%       param.gamma is used when computing the RBF kernel exp(-gamma*||x_i-x_j||);
%%  Reference:
%   Z.-H. Zhou, Y.-Y. Sun, and Y.-F. Li. Multi-instance learning by treating instances as non-i.i.d. samples. 
%   In: Proceedings of the 26th International Conference on Machine Learning (ICML'09), Montreal, Canada, 2009, pp.1249-1256.
%% End of Instruction


data1 = nargin{1,1};
data2 = nargin{2,1};
weight_cell1 = nargin{3,1};
weight_cell2 = nargin{4,1};
parameter = nargin{5,1};
gamma = parameter.gamma;


N1 = size(data1,1);
N2 = size(data2,1);

k = zeros(N1,N2);

tmp1 = zeros(N1,1);
tmp2 = zeros(N2,1);

for i = 1:N1
    tmp1(i) = Gau_kernel(data1{i,1}',data1{i,1}',weight_cell1{i,1},weight_cell1{i,1},gamma);
end

for i = 1:N2
    tmp2(i) = Gau_kernel(data2{i,1}',data2{i,1}',weight_cell2{i,1},weight_cell2{i,1},gamma);
end



for i = 1:N1
    for j = 1:N2
        k(i,j) = Gau_kernel(data1{i,1}',data2{j,1}',weight_cell1{i,1},weight_cell2{j,1},gamma)/sqrt(tmp1(i))/sqrt(tmp2(j));
        % kernel is normalized
    end
end

function k = Gau_kernel(bag1,bag2,weight1,weight2,gamma)
%%
%   This function is to compute the the RBF kernel
%%


count1 = sum(weight1);%count1(a)=\sum_{u=1}^{n_i} w_{au}^i
count2 = sum(weight2);% count2(b)=\sum_{v=1}^{n_j} w_{bv}^j

coef1 = 1./count1;%coef1(a)=1/count1(a)
coef2 = 1./count2;%coef2(b)=1/count2(b)

%%  Computing RBF kernel

N = size(bag1,2);
M = size(bag2,2);

zz = sum(bag1.^2);
xx = sum(bag2.^2);


K = repmat(zz',[1,M])+repmat(xx,[N,1])-2*bag1'*bag2;

clear zz;
clear xx;

K = exp(-gamma*K);

%%  End of computing kernel

%%  Computing k_g
K = repmat(coef1',[1,M]).*repmat(coef2,[N,1]).*K;
k = sum(sum(K))/sqrt(sum(coef1))/sqrt(sum(coef2));
%%  End of computing k_g

clear K;