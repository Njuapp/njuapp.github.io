# 多模态的文献阅读
## PMvGE
类别：
- 1.Graph Embedding
- 2.Multi-Modal Learning
  
问题：多对多关系、多模态数据下的子空间度量学习

算法：先用三层MLP抽取特征，然后内积的指数函数值作为相似度。相似度再乘一个模态相似性，作为两个节点之间这条边出现概率（泊松分布）的期望值，然后做最大似然估计，得出有效的MLP特征提取器。

技术：
- 1.最大似然估计
- 2.泊松分布

应用：
- 1.Clustering
- 2.Classification
- 3.Link Prediction

新意：现在只是加上了神经网络，用来近似非线性关系。还有概率化框架。

感悟：个人觉得，还是属于类label propagation的目标函数，即流形假设。只不过套了probabilistic的framework，使用了Poisson分布的MLE.

## RANC
