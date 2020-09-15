# 主成分分析PCA
## 背景
压缩数据维数，即位数约简。
<!-- ![Good PCA and Better PCA](D:\Course\机器学习\MachineLearning-FZU\image\PCA_GOOD_BETTER.jpg) -->
## 动机
- 数据在低维线性空间的正交投影：
  - 最大化投影数据的反差；
  - 最小化数据点与投影之间的均方距离。
## 架构
### 算法1
给定中心化数据$\{x_1,x_2,...,x_m\}$，计算主向量：
$w_1=arg\max\limits_{||w||=1}\frac{1}{m}\sum\limits_{i=1}^{m}\{(w^Tx_i^2)\}$
最大化x的投影方差
$w_k=arg\max\limits_{||w||=1}\frac{1}{m}\sum_{i=1}^{m}\{[w^T(x_i-\sum\limits_{j=1}^{k-1}w_jw_j^Tx_i)]^2\}$
## 应用