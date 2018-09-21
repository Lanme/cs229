## 因子分析

训练数据中样例$x_i$都远远大于其特征个数n，这样不管是进行回归、聚类等都没有太大的问题。然而当训练样例个数m太小，甚至m<<n的时候，使用梯度下降法进行回归时，如果初值不同，得到的参数结果会有很大偏差（因为方程数小于参数个数）。另外，如果使用多元高斯分布(Multivariate Gaussian distribution)对数据进行拟合时，也会有问题。 

1、考虑限制协方差矩阵

缺点：特征独立性的假设太强

2、因子分析

实质：数据降维

主成分分析分析与因子分析也有不同，主成分分析仅仅是变量变换，而因子分析需要构造因子模型。

主成分分析:原始变量的线性组合表示新的综合变量，即主成分。PCA只对符合高斯分布的样本点比较有效。

ICA：主元分解，不针对高斯分布。如经典的鸡尾酒问题。



因子分析：潜在的假想变量和随机影响变量的线性组合表示原始变量。



 因子分析的实质是认为m个n维特征的训练样例($x_1^{(i)},x_2^{(i)},...x_n^{(i)}$)的产生过程如下：

 1、 首先在一个k维的空间中按照多元高斯分布生成m个$z^{(i)}$(k维向量)，即
$$
z^{(i)} \sim N(0,I)
$$
2、然后存在一个变换矩阵 $\Lambda \in \R^{n \times k}$，将$z^{(i)}$ 映射到n维空间中，即
$$
\Lambda z^{(i)}
$$
因为$z^{(i)}$ 的均值是-，映射后仍然是0。

3、将$\Lambda z^{(i)}$将在一个均值$\mu$（n维）,维度与上面一致，即
$$
\mu + \Lambda z^{(i)}
$$
4、由于真实样例有误差，所以应该加上误差$\epsilon$ （n维），并且$\epsilon$符合高斯分布，即
$$
\epsilon \sim N(0,\Psi) \\
\mu + \Lambda z^{(i)} +\epsilon
$$
5、最后的结果即$x^{(i)}$的生成公式：
$$
x^{(i)} = \mu + \Lambda z^{(i)} +\epsilon
$$
![1](https://github.com/Lanme/cs229/raw/master/add_note/img/add_note9_1.png)

![2](https://github.com/Lanme/cs229/raw/master/add_note/img/add_note9_2.png)

![3](https://github.com/Lanme/cs229/raw/master/add_note/img/add_note9_3.png)

![4](https://github.com/Lanme/cs229/raw/master/add_note/img/add_note9_4.png)

![5](https://github.com/Lanme/cs229/raw/master/add_note/img/add_note9_5.png)

[参考地址]( https://blog.csdn.net/ruohuanni/article/details/42123625)

## 统计学的因子分析

- 因子旋转
- 因子载荷矩阵
- 

