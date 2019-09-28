------------------------------------------------------------------------------------------
	                   Readme for the miGraph Package
	 		       version July 4, 2009
------------------------------------------------------------------------------------------

The package includes the MATLAB code of the multi-instance learning algorithm miGraph, which does not assume the instance as i.i.d. samples [1].

[1] Z.-H. Zhou, Y.-Y. Sun, and Y.-F. Li. Multi-instance learning by treating instances as non-i.i.d. samples. In: Proceedings of the 26th International Conference on Machine Learning (ICML'09), Montreal, Canada, 2009, pp.1249-1256.


Note that the Matlab version of Libsvm is available at http://www.csie.ntu.edu.tw/~cjlin/libsvm/. It should be included here to facilitate the implementation of miGraph.

[2] C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector machines, Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan, Technical Report, 2001.


For miGraph, you can use the "miGraph" function to perform both training and testing process.

You will find an example of using this code in the example directory. The example data is alt.atheism. The fold of 10CV are also provided. 

In our ICML'09 experiments, C is fixed to 100, and $\gamma$ is selected via 5CV. 


ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other
  purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).

- This package was developed by Ms. Yu-Yin Sun (sunyy@lamda.nju.edu.cn). For any
  problem concerning the code, please feel free to contact Ms. Sun.

------------------------------------------------------------------------------------------