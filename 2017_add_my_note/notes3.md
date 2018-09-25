## 1、拉格朗日乘子法

> 转换为求最小值

1、无约束优化问题
$$
min f(x)
$$
2、有等式约束的优化问题
$$
min f(x) \\
s.t. h_i(x) = 0;i = 1,2,...
$$
3、有不等式约束的优化问题
$$
min f(x)\\
s.t. g_i(x) <=0,i=1,2,...\\
h_j(x)=0 ;j =1,2,...
$$


在SVM中的不等式约束为
$$
max_{w,b} \frac{1}{||w||} \\
s.t. y_i (w^T \cdot \Phi(x_i)+b)>=1 ；i=1,2,...N \\
$$
转为
$$
min_{w,b} \frac{1}{2}||w|| ^2 \\
s.t. y_i (w^T \cdot \Phi(x_i)+b)>=1 ；i=1,2,...N \\
$$
优化函数为
$$
L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^N\alpha_i (y_i (w^T \cdot \Phi(x_i)+b)-1)
$$
此时
$$
max_{\alpha} L(w,b,\alpha)
$$
为什么要这样假设呢？如果约束条件满足的话，最大值为$\frac{1}{2} ||w||$与原来的目标函数一直

原问题是极小极大问题
$$
\min_{w,b} \max_{\alpha} L(w,b,\alpha)
$$
对偶问题是极大极小问题
$$
\max_{\alpha} \min_{w,b} L(w,b,\alpha)
$$

## 2、原始问题与对偶问题的关系

 
$$
d \star = \max_{\alpha ,\beta ;\alpha_i \geq 0} \min_x L(x,\alpha,\beta) \\
\leq min_x max_{\alpha ,\beta ;\alpha_i \geq 0} L(x,\alpha , \beta) = p \star
$$
证明:
$$
\theta _D(\alpha,\beta) = min_x L(x,\alpha, \beta) \leq L(x,\alpha ,\beta) \\
\leq \max_{\alpha ,\beta ;\alpha_i \geq 0} L(x,\alpha , \beta) = \theta_p(X)
$$
即 
$$
\theta _D(\alpha,\beta) \leq \theta_p(x)
$$
由于原始问题与对偶问题都有最优值，所以
$$
\max_{\alpha ,\beta ;\alpha_i \geq 0} \theta _D(\alpha,\beta) \leq  min_{x} \theta_p(x)
$$

## 3、核函数

令X为输入空间，$\kappa(\cdot ,\cdot)$是定义在X*X上的对称函数，则$\kappa$是核函数当且仅当对于任意数据D = {$x_1，x_2,...x_m$}，核矩阵K总是半正定的:
$$
K = \left[\begin{matrix} 
\kappa(x_1,x_1) ...\kappa(x1,x_j )...\kappa(x_1,x_m) \\
... \\
\kappa(x_i,x_1) ...\kappa(xi,x_j )...\kappa(x_i,x_m) \\
... \\
\kappa(x_m,x_1) ...\kappa(x_m,x_j )...\kappa(x_m,x_m) \\
\end{matrix} \right]
$$
只要一个对称函数所对应的和矩阵半正定，它就能作为核函数使用。

常用的核函数

------

| 名称       | 表达式                                                       | 参数                     |
| ---------- | ------------------------------------------------------------ | ------------------------ |
| 线性核     | $\kappa(x_i,x_j)$ = $x^T_ix_j$                               |                          |
| 多项式核   | $\kappa(x_i,x_j)$= $(x_i^T x_j)^d$                           | d$\geq$1为多项式的次数   |
| 高斯核     | $\kappa(x_i,x_j)$ = exp($-\frac{||x_i - x_j||^2}{2 \sigma^2}$) | $\sigma>0$为高斯核的带宽 |
| 拉普拉斯核 | $\kappa(x_i,x_j)$ = exp(-$\frac{||x_i - x_j||^2}{\sigma}$)   | $\sigma>0$               |
| sigmod核   | $\kappa(x_i,x_j)$ = $\tan h( \beta x_i^T x_j + \theta)$      | $\beta$ >0 ,$\theta$ >0  |

核函数的选择，带入模型测试结果就知道了。

## 4、C值

C是惩罚因子，即对误差的宽容度。c越高，说明越不能容忍出现误差，容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差。

![2](https://github.com/Lanme/cs229/raw/master/add_note/img/add_notes3_2.png)

## 5、Gamma

对于RBF核函数来说
$$
\kappa (x,z) = exp(-gamma \cdot d(x,z)^2) \\
gamma = \frac{1}{2 \cdot \sigma^2}
$$
$\sigma$表示带宽，因此gamma越大 $\sigma$越小，正态分布越瘦高，使得靠近中间的向量权重更大。

![1](https://github.com/Lanme/cs229/raw/master/add_note/img/add_notes3_1.png)

gamma越大，靠近分割面的样本权重越大，导致分割面变得弯曲。

## 6、SMO

- 序列最小最优化

    Sequential Minimal Optimization

- 有多个拉格朗日乘子
- 每次只选择其中两个乘子做优化，其他因子认为是常数

