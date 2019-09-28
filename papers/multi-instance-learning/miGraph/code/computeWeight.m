function weight_cell = computeWeight(Bags,parameter)
%%  this function is used for computing weight
%   to use this function, LibSVM should be available.
%   refer: C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector
%   machines, Department of Computer Science and Information Engineering,
%   National Taiwan University, Taipei, Taiwan, Technical Report, 2001.
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou
%   (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Ms. Yu-Yin Sun (sunyy@lamda.nju.edu.cn). For any problem concerning the code,
%        please feel free to contact Ms. Sun.
%%  Reference:
%   Z.-H. Zhou, Y.-Y. Sun, and Y.-F. Li. Multi-instance learning by treating instances as non-i.i.d. samples. 
%   In: Proceedings of the 26th International Conference on Machine Learning (ICML'09), Montreal, Canada, 2009, pp.1249-1256.
%% End of Instruction


bag_num = size(Bags,1);
weight_cell = cell(bag_num,1);
for i = 1:bag_num
    inst_num = size(Bags{i,1},1);
    weight_cell{i,1} = zeros(inst_num,inst_num);
    zz = sum(Bags{i,1}.^2,2);
    weight_cell{i,1} = repmat(zz,[1,inst_num])+repmat(zz',[inst_num,1])-2*Bags{i,1}*Bags{i,1}';
    weight_cell{i,1} = double(weight_cell{i,1}<parameter.thr);
end